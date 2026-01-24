import os
import re
import torch
import numpy as np
import dgl
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

class VietnamesePreprocessor:
    def __init__(self):
        self.acronyms = {
            ':)': 'colonsmile',
            ':(': 'colonsad',
            '@@': 'colonsurprise',
            '<3': 'colonlove',
            ':d': 'colonsmilesmile',
            ':3': 'coloncontemn',
            ':v': 'colonbigsmile',
            ':_': 'coloncc',
            ':p': 'colonsmallsmile',
            '>>': 'coloncolon',
            ':">': 'colonlovelove',
            '^^': 'colonhihi',
            ':': 'doubledot',
            ":'(": 'colonsadcolon',
            ':@': 'colondoublesurprise',
            'v.v': 'vdotv',
            '...': 'dotdotdot',
            '/': 'fraction',
            'c#': 'cshrap'
        }
        
        self.stopwords = set([
            'và', 'của', 'có', 'là', 'được', 'cho', 'với', 'từ', 'trong',
            'này', 'đó', 'để', 'một', 'những', 'các', 'không', 'thì', 'như',
            'đã', 'sẽ', 'khi', 'nếu', 'vì', 'mà', 'về', 'hoặc', 'hay',
            'đến', 'bởi', 'tại', 'theo', 'nên', 'nhưng', 'chỉ', 'nào', 'đây',
            'đấy', 'ở', 'ra', 'vào', 'lại', 'còn', 'cũng', 'rất', 'đều'
        ])

    def replace_acronyms(self, text):
        sorted_acronyms = sorted(self.acronyms.items(), 
                                key=lambda x: len(x[0]), 
                                reverse=True)
        
        for acronym, replacement in sorted_acronyms:
            text = text.replace(acronym, f' {replacement} ')
        
        return text
    
    def preprocess(self, text, remove_stopwords=False):
        text = self.replace_acronyms(text)
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
    
        tokens = text.split()
        tokens = [t.strip() for t in tokens if t.strip()]
        
        if remove_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]
        
        return tokens
    
    def preprocess_corpus(self, texts, remove_stopwords=False):
        return [self.preprocess(text, remove_stopwords) for text in texts]

class CustomDatasetLoader:
    def __init__(self, data_dir='./data'):
        self.data_dir = data_dir
        self.preprocessor = VietnamesePreprocessor()
    
    def load_split(self, split_name):
        split_dir = os.path.join(self.data_dir, split_name)
        
        with open(os.path.join(split_dir, 'sents.txt'), 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f]

        with open(os.path.join(split_dir, 'sentiments.txt'), 'r', encoding='utf-8') as f:
            sentiments = [int(line.strip()) for line in f]
    
        with open(os.path.join(split_dir, 'topics.txt'), 'r', encoding='utf-8') as f:
            topics = [int(line.strip()) for line in f]
        
        assert len(texts) == len(sentiments) == len(topics)
        
        return texts, sentiments, topics
    
    def load_all_splits(self, task='sentiment', remove_stopwords=False):
        train_texts, train_sents, train_topics = self.load_split('train')
        dev_texts, dev_sents, dev_topics = self.load_split('dev')
        test_texts, test_sents, test_topics = self.load_split('test')
        
        train_docs = self.preprocessor.preprocess_corpus(train_texts, remove_stopwords)
        dev_docs = self.preprocessor.preprocess_corpus(dev_texts, remove_stopwords)
        test_docs = self.preprocessor.preprocess_corpus(test_texts, remove_stopwords)
        
        if task == 'sentiment':
            train_labels, dev_labels, test_labels = train_sents, dev_sents, test_sents
            num_classes = 3
        elif task == 'topic':
            train_labels, dev_labels, test_labels = train_topics, dev_topics, test_topics
            num_classes = 4
        else:
            raise ValueError(f"Not found: {task}")
        
        def avg_len(docs):
            return sum(len(d) for d in docs) / len(docs) if docs else 0

        print(f"\nLabel Distribution:")
        for split_name, labels in [('Train', train_labels), ('Dev', dev_labels), ('Test', test_labels)]:
            counts = defaultdict(int)
            for label in labels:
                counts[label] += 1
        
        return (train_docs, train_labels, dev_docs, dev_labels, 
                test_docs, test_labels, num_classes)


def load_glove_embeddings(vocab, embedding_dim=300, glove_path=None):
    vocab_size = len(vocab)
    
    embeddings = np.random.randn(vocab_size, embedding_dim).astype(np.float32)
    embeddings = embeddings * 0.01
    
    if glove_path and os.path.exists(glove_path):
        print(f"Loading embeddings from {glove_path}...")
        found = 0
        
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.strip().split()
                if len(values) < embedding_dim + 1:
                    continue
                    
                word = values[0]
                if word in vocab:
                    vector = np.array(values[1:], dtype=np.float32)
                    if len(vector) == embedding_dim:
                        embeddings[vocab[word]] = vector
                        found += 1
        
        print(f" Found {found}/{vocab_size} words in GloVe.")
    else:
        print(f"GloVe path not provided or does not exist. Using random embeddings.")
    
    return embeddings


class GFNDataset(Dataset):    
    def __init__(self, documents, labels, graph_constructor, corpus_graphs, 
                 p_neighbors=None):
        self.documents = documents
        self.labels = labels
        self.graph_constructor = graph_constructor
        self.corpus_graphs = corpus_graphs
        self.p_neighbors = p_neighbors
    
    def __len__(self):
        return len(self.documents)
    
    def __getitem__(self, idx):
        doc = self.documents[idx]
        label = self.labels[idx]
        
        subgraphs = []
        node_indices_list = []
        edge_weights_list = []
        
        for corpus_adj in self.corpus_graphs:
            subgraph_adj, word_indices = self.graph_constructor.build_document_subgraph( doc, corpus_adj, p_neighbors=self.p_neighbors)
            
            if subgraph_adj is None or len(word_indices) == 0:
                subgraph_adj = np.array([[0.0]], dtype=np.float32)
                word_indices = [0]  
 
            src, dst = np.nonzero(subgraph_adj)
            
            if len(src) == 0:
                src = np.array([0])
                dst = np.array([0])
                edge_weights = np.array([1.0], dtype=np.float32)
            else:
                edge_weights = subgraph_adj[src, dst].astype(np.float32)
            
            g = dgl.graph((src, dst), num_nodes=len(word_indices))
            
            subgraphs.append(g)
            node_indices_list.append(torch.LongTensor(word_indices))
            edge_weights_list.append(torch.FloatTensor(edge_weights))
        
        return subgraphs, node_indices_list, edge_weights_list, label


def collate_fn(batch):
    all_subgraphs = [item[0] for item in batch]
    all_node_indices = [item[1] for item in batch]
    all_edge_weights = [item[2] for item in batch]
    labels = torch.LongTensor([item[3] for item in batch])
    
    batched_graphs = []
    batched_node_indices = []
    batched_edge_weights = []
    
    num_graphs = len(all_subgraphs[0]) 
    
    for graph_idx in range(num_graphs):
        graphs = [sample[graph_idx] for sample in all_subgraphs]
        node_indices = [sample[graph_idx] for sample in all_node_indices]
        edge_weights = [sample[graph_idx] for sample in all_edge_weights]
        
        batched_graph = dgl.batch(graphs)
        batched_node_idx = torch.cat(node_indices, dim=0)
        batched_edge_weight = torch.cat(edge_weights, dim=0)
        batched_graphs.append(batched_graph)
        batched_node_indices.append(batched_node_idx)
        batched_edge_weights.append(batched_edge_weight)
    
    return batched_graphs, batched_node_indices, batched_edge_weights, labels


def create_data_loaders(train_docs, train_labels, dev_docs, dev_labels, test_docs, test_labels, graph_constructor, corpus_graphs, batch_size=32, p_neighbors=None):

    train_dataset = GFNDataset(train_docs, train_labels, graph_constructor, corpus_graphs, p_neighbors)
    dev_dataset = GFNDataset(dev_docs, dev_labels, graph_constructor, corpus_graphs, p_neighbors)
    test_dataset = GFNDataset(test_docs, test_labels, graph_constructor, corpus_graphs, p_neighbors)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, dev_loader, test_loader
