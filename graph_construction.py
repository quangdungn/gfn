import numpy as np
from collections import defaultdict
from scipy.sparse import lil_matrix
import warnings
warnings.filterwarnings('ignore')


class GraphConstructor:
    def __init__(self, window_size=20):
        self.window_size = window_size
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0
        
    def build_vocabulary(self, documents, min_freq=1):
        word_freq = defaultdict(int)
        for doc in documents:
            for word in doc:
                word_freq[word] += 1
        
        vocab = [word for word, freq in word_freq.items() if freq >= min_freq]
        
        self.word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_size = len(self.word_to_idx)
        
        print(f"Word si \: {self.vocab_size} (min_freq={min_freq})")
        
        return self.word_to_idx
    
    def compute_cooccurrence_statistics(self, documents):
        n = self.vocab_size
        cooc_count = lil_matrix((n, n), dtype=np.float32)
        word_count = np.zeros(n, dtype=np.float32)

        for doc in documents:
            indices = [self.word_to_idx[w] for w in doc if w in self.word_to_idx]
        
            for idx in indices:
                word_count[idx] += 1

            for i, center_idx in enumerate(indices):
                start = max(0, i - self.window_size)
                end = min(len(indices), i + self.window_size + 1)
                
                for j in range(start, end):
                    if i != j:
                        context_idx = indices[j]
                        cooc_count[center_idx, context_idx] += 1
        
        return cooc_count.toarray(), word_count
    
    def build_graph_cooccurrence(self, documents):
        cooc_count, word_count = self.compute_cooccurrence_statistics(documents)
        n = self.vocab_size
    
        adjacency = np.zeros((n, n), dtype=np.float32)
        
        for i in range(n):
            if word_count[i] > 0:
                adjacency[i] = cooc_count[i] / word_count[i]
        
        np.fill_diagonal(adjacency, 0)
        
        num_edges = np.count_nonzero(adjacency)
        return adjacency
    
    def build_graph_ppmi(self, documents):
        cooc_count, word_count = self.compute_cooccurrence_statistics(documents)
        total_pairs = np.sum(cooc_count)
        denominator = np.outer(word_count, word_count)
        adjacency = np.zeros_like(cooc_count, dtype=np.float32)

        valid = (cooc_count > 0) & (denominator > 0)
        if np.any(valid) and total_pairs > 0:
            pmi = np.log((cooc_count[valid] * total_pairs) / denominator[valid])
            adjacency[valid] = np.maximum(pmi, 0.0).astype(np.float32)

        np.fill_diagonal(adjacency, 0)
        return adjacency
    
    def build_graph_cosine(self, embeddings):
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / np.clip(norms, 1e-12, None)
        adjacency = normalized @ normalized.T
        adjacency = np.maximum(adjacency, 0.0).astype(np.float32)

        np.fill_diagonal(adjacency, 0)
        return adjacency
    
    def build_graph_euclidean(self, embeddings):   
        squared_norms = np.sum(embeddings ** 2, axis=1, keepdims=True)
        distances_sq = squared_norms + squared_norms.T - 2.0 * (embeddings @ embeddings.T)
        distances_sq = np.maximum(distances_sq, 0.0)
        distances = np.sqrt(distances_sq)
        adjacency = (1.0 / (1.0 + distances)).astype(np.float32)
        
        np.fill_diagonal(adjacency, 0)
        return adjacency
    
    def filter_edges(self, adjacency, keep_top_k=None):
        if keep_top_k is None:
            keep_top_k = min(int(0.005 * self.vocab_size), 100)
        
        n = adjacency.shape[0]
        filtered = np.zeros_like(adjacency)
        
        for i in range(n):
            weights = adjacency[i].copy()
            
            if np.count_nonzero(weights) > keep_top_k:
                top_k_indices = np.argpartition(weights, -keep_top_k)[-keep_top_k:]
                filtered[i, top_k_indices] = weights[top_k_indices]
            else:
                filtered[i] = weights
        
        return filtered
    
    def build_all_graphs(self, documents, embeddings, filter_edges=True):
        A1 = self.build_graph_cooccurrence(documents)
        A2 = self.build_graph_ppmi(documents)
        A3 = self.build_graph_cosine(embeddings)
        A4 = self.build_graph_euclidean(embeddings)
        
        graphs = [A1, A2, A3, A4]
        
        if filter_edges:
            keep_top_k = min(int(0.005 * self.vocab_size), 100)
            
            filtered_graphs = []
            for i, graph in enumerate(graphs):
                filtered = self.filter_edges(graph, keep_top_k)
                num_edges = np.count_nonzero(filtered)
                filtered_graphs.append(filtered)
            
            graphs = filtered_graphs
        
        return graphs
    
    def build_document_subgraph(self, doc_words, corpus_adjacency, p_neighbors=None):
        word_indices = []
        for word in doc_words:
            if word in self.word_to_idx:
                idx = self.word_to_idx[word]
                if idx not in word_indices:
                    word_indices.append(idx)
        
        if len(word_indices) == 0:
            return None, None
        
        n_doc = len(word_indices)
        subgraph_adj = np.zeros((n_doc, n_doc), dtype=np.float32)
        
        global_to_local = {global_idx: local_idx 
                          for local_idx, global_idx in enumerate(word_indices)}
        
        for i, wi in enumerate(word_indices):
            for j, wj in enumerate(word_indices):
                subgraph_adj[i, j] = corpus_adjacency[wi, wj]
        
        if p_neighbors is not None and p_neighbors > 0:
            for i, global_idx in enumerate(word_indices):
                neighbors = corpus_adjacency[global_idx]
                
                neighbor_indices = np.argsort(neighbors)[-p_neighbors:]
                
                for neighbor_idx in neighbor_indices:
                    if neighbor_idx in global_to_local:
                        j = global_to_local[neighbor_idx]
                        subgraph_adj[i, j] = max(subgraph_adj[i, j], 
                                                neighbors[neighbor_idx])
        
        return subgraph_adj, word_indices
