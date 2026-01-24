import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import argparse
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report

class GFNTrainer:
    def __init__(self, model, device, config):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        
    def train_single_graph_epoch(self, graph_idx, train_loader, optimizer):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in train_loader:
            graphs_batch, node_indices_batch, edge_weights_batch, labels = batch

            labels = labels.to(self.device)
            g = graphs_batch[graph_idx].to(self.device)
            node_indices = node_indices_batch[graph_idx].to(self.device)
            edge_weights = edge_weights_batch[graph_idx].to(self.device)
            optimizer.zero_grad()

            node_features = self.model.embeddings(node_indices)
            final_features = self.model.graph_convolution( g, node_features, edge_weights, graph_idx)
            doc_emb = self.model.document_embedding(g, final_features)
            logits = self.model.classify(doc_emb, graph_idx)
            loss = self.criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    def evaluate_single_graph(self, graph_idx, data_loader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                graphs_batch, node_indices_batch, edge_weights_batch, labels = batch
                
                labels = labels.to(self.device)
                g = graphs_batch[graph_idx].to(self.device)
                node_indices = node_indices_batch[graph_idx].to(self.device)
                edge_weights = edge_weights_batch[graph_idx].to(self.device)
                
                node_features = self.model.embeddings(node_indices)
                final_features = self.model.graph_convolution(
                    g, node_features, edge_weights, graph_idx
                )
                doc_emb = self.model.document_embedding(g, final_features)
                logits = self.model.classify(doc_emb, graph_idx)
                
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    def train_stage1(self, train_loader, dev_loader):
        for graph_idx in range(4):
            print(f"\n{'='*60}")
            print(f"Training Graph {graph_idx + 1}/4")
            print(f"{'='*60}")
            
            params = list(self.model.input_projection.parameters())
            params += [self.model.edge_learning_params[graph_idx]] 
            params += list(self.model.classifiers[graph_idx].parameters())
            
            optimizer = torch.optim.AdamW(params, lr=self.config['learning_rate'])
            
            best_val_loss = float('inf')
            patience_counter = 0
            patience = self.config['stage1_patience'] 
            
            for epoch in range(self.config['stage1_epochs']):
                train_loss, train_acc = self.train_single_graph_epoch( graph_idx, train_loader, optimizer)

                val_loss, val_acc = self.evaluate_single_graph( graph_idx, dev_loader)
                
                print(f"Epoch {epoch+1:3d}: "
                      f"Train Loss={train_loss:.4f} Train Acc={train_acc:.4f} | "
                      f"Val Loss={val_loss:.4f} Val Acc={val_acc:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save_graph_state(graph_idx)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            
            self.load_graph_state(graph_idx)
    
    def train_stage2_epoch(self, train_loader, optimizer):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in train_loader:
            graphs_batch, node_indices_batch, edge_weights_batch, labels = batch
            
            labels = labels.to(self.device)
            graphs_batch = [g.to(self.device) for g in graphs_batch]
            node_indices_batch = [n.to(self.device) for n in node_indices_batch]
            edge_weights_batch = [e.to(self.device) for e in edge_weights_batch]
            
            optimizer.zero_grad()
            predictions, fused_logits = self.model(
                graphs_batch, node_indices_batch, edge_weights_batch
            )
            
            loss = self.criterion(fused_logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(fused_logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    def train_stage2(self, train_loader, dev_loader):      
        for param in self.model.input_projection.parameters():
            param.requires_grad = False
        for param in self.model.edge_learning_params.parameters():
            param.requires_grad = False
        for param in self.model.classifiers.parameters():
            param.requires_grad = False
        for param in self.model.embeddings.parameters():
            param.requires_grad = False
        
        optimizer = torch.optim.AdamW(
            self.model.fusion_heads.parameters(),
            lr=self.config['fusion_lr']  
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = self.config['stage2_patience'] 
        
        iteration = 0
        max_iterations = self.config['stage2_iterations']
        
        while iteration < max_iterations:
            train_loss, train_acc = self.train_stage2_epoch(train_loader, optimizer)
            val_loss, val_acc, val_micro_f1, val_macro_f1 = self.evaluate(dev_loader)
            iteration += 1
            
            print(f"Iteration {iteration:4d}: "
                  f"Train Loss={train_loss:.4f} Train Acc={train_acc:.4f} | "
                  f"Val Loss={val_loss:.4f} Val Acc={val_acc:.4f} "
                  f"Micro-F1={val_micro_f1:.4f} Macro-F1={val_macro_f1:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model('best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at iteration {iteration}")
                    break
        
        self.load_model('best_model.pth')
    
    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                graphs_batch, node_indices_batch, edge_weights_batch, labels = batch
        
                labels = labels.to(self.device)
                graphs_batch = [g.to(self.device) for g in graphs_batch]
                node_indices_batch = [n.to(self.device) for n in node_indices_batch]
                edge_weights_batch = [e.to(self.device) for e in edge_weights_batch]

                predictions, fused_logits = self.model(
                    graphs_batch, node_indices_batch, edge_weights_batch
                )

                loss = self.criterion(fused_logits, labels)
                total_loss += loss.item()

                preds = torch.argmax(fused_logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        micro_f1 = f1_score(all_labels, all_preds, average='micro')
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        
        return avg_loss, accuracy, micro_f1, macro_f1
    
    def save_model(self, filename):
        path = os.path.join(self.config['save_dir'], filename)
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, filename):
        path = os.path.join(self.config['save_dir'], filename)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
    
    def save_graph_state(self, graph_idx):
        path = os.path.join(self.config['save_dir'], f'graph_{graph_idx}_best.pth')
        state = {
            'input_projection': self.model.input_projection.state_dict(),
            'edge_learning': self.model.edge_learning_params[graph_idx].data.clone(),
            'classifier': self.model.classifiers[graph_idx].state_dict()
        }
        torch.save(state, path)
    
    def load_graph_state(self, graph_idx):
        path = os.path.join(self.config['save_dir'], f'graph_{graph_idx}_best.pth')
        if os.path.exists(path):
            state = torch.load(path, map_location=self.device)
            self.model.input_projection.load_state_dict(state['input_projection'])
            self.model.edge_learning_params[graph_idx].data.copy_(state['edge_learning'])
            self.model.classifiers[graph_idx].load_state_dict(state['classifier'])

def run_training_pipeline(config):
    from gfn_custom_dataset import (CustomDatasetLoader, load_glove_embeddings, create_data_loaders)
    from graph_construction import GraphConstructor
    from gfn_model import GraphFusionNetwork
    
    loader = CustomDatasetLoader(data_dir=config['data_dir'])
    (train_docs, train_labels, dev_docs, dev_labels, test_docs, test_labels, num_classes) = loader.load_all_splits(task=config['task'], remove_stopwords=config['remove_stopwords'])
    
    all_docs = train_docs + dev_docs + test_docs
    
    constructor = GraphConstructor(window_size=config['window_size'])
    vocab = constructor.build_vocabulary(all_docs, min_freq=config['min_freq'])
    embeddings = load_glove_embeddings(
        vocab, 
        embedding_dim=config['embedding_dim'],
        glove_path=config.get('glove_path')
    )
    corpus_graphs = constructor.build_all_graphs(
        all_docs, embeddings, filter_edges=True
    )

    train_loader, dev_loader, test_loader = create_data_loaders(
        train_docs, train_labels, dev_docs, dev_labels,
        test_docs, test_labels, constructor, corpus_graphs,
        batch_size=config['batch_size'],
        p_neighbors=config.get('p_neighbors')
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Dev batches: {len(dev_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    model = GraphFusionNetwork(
        vocab_size=len(vocab),
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_classes=num_classes,
        num_graphs=4,
        num_heads=config['num_heads'],
        num_conv_steps=2, 
        dropout=config['dropout']
    )
    model.load_pretrained_embeddings(embeddings)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    os.makedirs(config['save_dir'], exist_ok=True)
    
    device = torch.device(config['device'])
    trainer = GFNTrainer(model, device, config)
    
    trainer.train_stage1(train_loader, dev_loader)
    trainer.train_stage2(train_loader, dev_loader)
    
    test_loss, test_acc, test_micro_f1, test_macro_f1 = trainer.evaluate(test_loader)
    
    print(f"\nTest Results:")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Micro-F1:  {test_micro_f1:.4f}")
    print(f"  Macro-F1:  {test_macro_f1:.4f}")
    
    results = {
        'task': config['task'],
        'test_accuracy': float(test_acc),
        'test_micro_f1': float(test_micro_f1),
        'test_macro_f1': float(test_macro_f1),
        'config': config
    }
    
    results_path = os.path.join(config['save_dir'], 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description='GFN Training Pipeline for Vietnamese Student Feedback\'s Corpus'
    )
    
    parser.add_argument('--task', type=str, default='sentiment', choices=['sentiment', 'topic'])
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--window_size', type=int, default=20)
    parser.add_argument('--min_freq', type=int, default=1)
    parser.add_argument('--embedding_dim', type=int, default=300)
    parser.add_argument('--hidden_dim', type=int, default=300)
    parser.add_argument('--num_heads', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--p_neighbors', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--fusion_lr', type=float, default=0.05)
    parser.add_argument('--stage1_epochs', type=int, default=100)
    parser.add_argument('--stage1_patience', type=int, default=10)
    parser.add_argument('--stage2_iterations', type=int, default=1000)
    parser.add_argument('--stage2_patience', type=int, default=100)
    parser.add_argument('--remove_stopwords', action='store_true')
    parser.add_argument('--glove_path', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    config = vars(args)
    
    results = run_training_pipeline(config)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)

if __name__ == "__main__":
    main()