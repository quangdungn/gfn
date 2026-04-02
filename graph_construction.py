import numpy as np
from collections import defaultdict
from scipy.sparse import lil_matrix
import warnings
warnings.filterwarnings('ignore')


class GraphConstructor:
    def __init__(
        self,
        window_size=20,
        sequential_window=1,
        sequential_edge_weight=0.25,
        preserve_multiplicity=True,
        unk_token="<unk>",
    ):
        self.window_size = window_size
        self.sequential_window = sequential_window
        self.sequential_edge_weight = sequential_edge_weight
        self.preserve_multiplicity = preserve_multiplicity
        self.unk_token = unk_token
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0
        
    def build_vocabulary(self, documents, min_freq=1):
        word_freq = defaultdict(int)
        for doc in documents:
            for word in doc:
                word_freq[word] += 1
        
        vocab = [self.unk_token]
        vocab.extend(
            word
            for word, freq in word_freq.items()
            if freq >= min_freq and word != self.unk_token
        )
        
        self.word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_size = len(self.word_to_idx)
        
        print(f"Vocabulary size: {self.vocab_size} (min_freq={min_freq})")
        
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
            keep_top_k = max(1, min(int(0.005 * self.vocab_size), 100))
        
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
            keep_top_k = max(1, min(int(0.005 * self.vocab_size), 100))
            
            filtered_graphs = []
            for i, graph in enumerate(graphs):
                filtered = self.filter_edges(graph, keep_top_k)
                num_edges = np.count_nonzero(filtered)
                filtered_graphs.append(filtered)
            
            graphs = filtered_graphs
        
        return graphs
    
    def build_document_subgraph(self, doc_words, corpus_adjacency, p_neighbors=None):
        unk_idx = self.word_to_idx.get(self.unk_token)
        if self.preserve_multiplicity:
            word_indices = [
                self.word_to_idx.get(word, unk_idx)
                for word in doc_words
                if self.word_to_idx.get(word, unk_idx) is not None
            ]
        else:
            seen = set()
            word_indices = []
            for word in doc_words:
                idx = self.word_to_idx.get(word, unk_idx)
                if idx is None or idx in seen:
                    continue
                seen.add(idx)
                word_indices.append(idx)
        
        if len(word_indices) == 0:
            return None, None
        
        n_doc = len(word_indices)
        subgraph_adj = corpus_adjacency[np.ix_(word_indices, word_indices)].astype(np.float32)

        global_to_local = defaultdict(list)
        for local_idx, global_idx in enumerate(word_indices):
            global_to_local[global_idx].append(local_idx)

        if self.sequential_window > 0 and self.sequential_edge_weight > 0:
            for i in range(n_doc):
                max_j = min(n_doc, i + self.sequential_window + 1)
                for j in range(i + 1, max_j):
                    local_weight = self.sequential_edge_weight / float(j - i)
                    subgraph_adj[i, j] = max(subgraph_adj[i, j], local_weight)
                    subgraph_adj[j, i] = max(subgraph_adj[j, i], local_weight)
        
        if p_neighbors is not None and p_neighbors > 0:
            for i, global_idx in enumerate(word_indices):
                neighbors = corpus_adjacency[global_idx]
                
                positive_neighbors = np.flatnonzero(neighbors > 0)
                if positive_neighbors.size == 0:
                    continue

                if positive_neighbors.size > p_neighbors:
                    top_positions = np.argpartition(neighbors[positive_neighbors], -p_neighbors)[-p_neighbors:]
                    neighbor_indices = positive_neighbors[top_positions]
                else:
                    neighbor_indices = positive_neighbors

                for neighbor_idx in neighbor_indices:
                    for j in global_to_local.get(neighbor_idx, []):
                        if i == j:
                            continue
                        subgraph_adj[i, j] = max(subgraph_adj[i, j], neighbors[neighbor_idx])

        return subgraph_adj, word_indices
