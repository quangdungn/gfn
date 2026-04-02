import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np


class GraphFusionNetwork(nn.Module):    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_graphs=4, num_heads=3, num_conv_steps=2, dropout=0.5):
        super(GraphFusionNetwork, self).__init__()
        
        assert num_graphs == 4
    
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_graphs = num_graphs
        self.num_heads = num_heads
        self.num_conv_steps = num_conv_steps
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.edge_learning_params = nn.ParameterList([ nn.Parameter(torch.ones(1)) for _ in range(num_graphs)])
        self.input_projections = nn.ModuleList(
            [nn.Linear(embedding_dim, hidden_dim) for _ in range(num_graphs)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifiers = nn.ModuleList([nn.Linear(hidden_dim, num_classes) for _ in range(num_graphs)])
        self.fusion_heads = nn.ModuleList([
            nn.Conv1d(
                in_channels=num_classes,
                out_channels=num_classes,
                kernel_size=num_graphs,
                groups=1
            ) for _ in range(num_heads)
        ])
        
    def load_pretrained_embeddings(self, pretrained_embeddings):
        self.embeddings.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
    
    def graph_learning(self, edge_weights, graph_idx):
        adjusted = F.relu(self.edge_learning_params[graph_idx] * edge_weights)
        return adjusted
    
    def message_passing_step(self, graph, node_features, edge_weights):
        with graph.local_scope():
            graph.ndata['h'] = node_features
            graph.edata['w'] = edge_weights.unsqueeze(-1) 
            graph.update_all(
                message_func=dgl.function.u_mul_e('h', 'w', 'm'),
                reduce_func=dgl.function.mean('m', 'agg')
            )

            new_features = node_features + graph.ndata['agg']
            
            return new_features
    
    def graph_convolution(self, graph, node_features, edge_weights, graph_idx):
        h = self.input_projections[graph_idx](node_features)

        adjusted_edge_weights = self.graph_learning(edge_weights, graph_idx)

        for t in range(self.num_conv_steps):
            h = self.message_passing_step(graph, h, adjusted_edge_weights)
        
        return h
    
    def document_embedding(self, graph, final_node_features):

        with graph.local_scope():
            graph.ndata['h'] = final_node_features
            doc_emb = dgl.mean_nodes(graph, 'h')
            return doc_emb
    
    def classify(self, doc_embedding, graph_idx):
        doc_emb_activated = self.dropout(F.relu(doc_embedding))
        logits = self.classifiers[graph_idx](doc_emb_activated)
        return logits
    
    def multi_head_fusion(self, logits_list):
        stacked_logits = torch.stack(logits_list, dim=-1)
        head_outputs = []
        for head in self.fusion_heads:
            head_out = head(stacked_logits).squeeze(-1) 
            head_outputs.append(head_out)

        fused_logits = torch.mean(torch.stack(head_outputs, dim=0), dim=0)
        
        return fused_logits
    
    def forward(self, graphs_batch, node_indices_batch, edge_weights_batch):
        logits_list = []
        
        for graph_idx in range(self.num_graphs):
            graph = graphs_batch[graph_idx]
            node_indices = node_indices_batch[graph_idx]
            edge_weights = edge_weights_batch[graph_idx]
            
            node_features = self.embeddings(node_indices)  

            final_node_features = self.graph_convolution(
                graph, node_features, edge_weights, graph_idx
            )
            
            doc_embedding = self.document_embedding(graph, final_node_features)
            
            logits = self.classify(doc_embedding, graph_idx)
            
            logits_list.append(logits)

        fused_logits = self.multi_head_fusion(logits_list)
        predictions = F.softmax(fused_logits, dim=-1)
        
        return predictions, fused_logits


class GFNLoss(nn.Module):
    def __init__(self):
        super(GFNLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, logits, labels):
        return self.criterion(logits, labels)
