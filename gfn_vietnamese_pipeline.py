import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import argparse
import warnings
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report

EMBEDDING_PRESETS = {
    'none': None,
    'phow2v_syllables_300': {
        'embedding_dim': 300,
        'search_tokens': ['phow2v', 'syllables', '300'],
        'search_filenames': [
            'PhoW2V_syllables_300dims.txt',
            'PhoW2V_syllables_300dims.vec',
            'PhoW2V_syllables_300dims.zip',
            'word2vec_vi_syllables_300dims.zip',
        ],
    },
    'phow2v_words_300': {
        'embedding_dim': 300,
        'search_tokens': ['phow2v', 'words', '300'],
        'search_filenames': [
            'PhoW2V_words_300dims.txt',
            'PhoW2V_words_300dims.vec',
            'PhoW2V_words_300dims.zip',
            'word2vec_vi_words_300dims.zip',
        ],
    },
}


def resolve_data_dir(data_dir):
    raw_candidates = []
    if data_dir:
        raw_candidates.append(data_dir)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    if data_dir and not os.path.isabs(data_dir):
        raw_candidates.append(os.path.join(os.getcwd(), data_dir))
        raw_candidates.append(os.path.join(script_dir, data_dir))
        raw_candidates.append(os.path.join(project_root, data_dir))

    raw_candidates.extend([
        os.path.join(os.getcwd(), 'dataGPT'),
        os.path.join(script_dir, 'dataGPT'),
        os.path.join(project_root, 'dataGPT'),
    ])

    candidates = []
    for candidate in raw_candidates:
        normalized = os.path.abspath(candidate)
        if normalized not in candidates:
            candidates.append(normalized)

    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate

    searched = "\n  - ".join(candidates)
    raise FileNotFoundError(
        "Could not find a valid data directory. Checked:\n"
        f"  - {searched}"
    )


def resolve_device(requested_device):
    requested_device = (requested_device or 'auto').strip()
    requested_lower = requested_device.lower()

    if requested_lower == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if requested_lower.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is False. "
                "Please install a CUDA-enabled PyTorch build, or run with --device cpu."
            )
        device = torch.device(requested_device)
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {device.index} is out of range. "
                f"Found {torch.cuda.device_count()} CUDA device(s)."
            )
        return device

    return torch.device(requested_device)


def validate_graph_device(device):
    if device.type != 'cuda':
        return

    try:
        import dgl

        dgl.graph(([0], [0]), num_nodes=1).to(device)
    except Exception as exc:
        raise RuntimeError(
            "CUDA is visible to PyTorch, but DGL could not move a graph to the GPU. "
            "Install a CUDA-enabled DGL build that matches your PyTorch/CUDA version."
        ) from exc


def infer_loader_config(config, device):
    num_workers = config.get('num_workers')
    if num_workers is None:
        cpu_count = os.cpu_count() or 0
        if os.name == 'nt':
            num_workers = 0
        else:
            num_workers = min(4, cpu_count) if device.type == 'cuda' else 0

    pin_memory = config.get('pin_memory')
    if pin_memory is None:
        pin_memory = device.type == 'cuda'

    persistent_workers = config.get('persistent_workers')
    if persistent_workers is None:
        persistent_workers = num_workers > 0

    return {
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'persistent_workers': persistent_workers,
    }


def resolve_embedding_path(config):
    explicit_path = config.get('embedding_path') or config.get('glove_path')
    if explicit_path:
        resolved = os.path.abspath(explicit_path)
        if os.path.exists(resolved):
            return resolved
        raise FileNotFoundError(f"Embedding path does not exist: {resolved}")

    preset_name = config.get('embedding_preset', 'none')
    preset = EMBEDDING_PRESETS.get(preset_name)
    if not preset:
        return None

    if config['embedding_dim'] != preset['embedding_dim']:
        raise ValueError(
            f"Preset '{preset_name}' expects embedding_dim={preset['embedding_dim']}, "
            f"but got {config['embedding_dim']}."
        )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    search_roots = [
        os.path.join(script_dir, 'embeddings'),
        os.path.join(project_root, 'embeddings'),
        os.path.join(project_root, 'gfn', 'embeddings'),
        os.path.join(os.getcwd(), 'gfn', 'embeddings'),
        os.path.join(os.getcwd(), 'embeddings'),
    ]

    candidates = []
    for root in search_roots:
        if not os.path.isdir(root):
            continue

        for filename in preset['search_filenames']:
            candidate = os.path.join(root, filename)
            if os.path.exists(candidate):
                return os.path.abspath(candidate)

        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                lower_name = filename.lower()
                if not lower_name.endswith(('.txt', '.vec', '.zip', '.gz', '.emb')):
                    continue
                if all(token in lower_name for token in preset['search_tokens']):
                    candidates.append(os.path.join(dirpath, filename))

    if candidates:
        candidates.sort()
        return os.path.abspath(candidates[0])

    print(
        f"Embedding preset '{preset_name}' was requested, but no local file was found. "
        "Falling back to random embeddings."
    )
    return None


class GFNTrainer:
    def __init__(self, model, device, config):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        self.non_blocking = device.type == 'cuda'
        self.use_amp = bool(config.get('amp', False) and device.type == 'cuda')
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def _move_single_graph_batch(self, batch, graph_idx):
        graphs_batch, node_indices_batch, edge_weights_batch, labels = batch

        labels = labels.to(self.device, non_blocking=self.non_blocking)
        graph = graphs_batch[graph_idx].to(self.device)
        node_indices = node_indices_batch[graph_idx].to(
            self.device, non_blocking=self.non_blocking
        )
        edge_weights = edge_weights_batch[graph_idx].to(
            self.device, non_blocking=self.non_blocking
        )
        return graph, node_indices, edge_weights, labels

    def _move_full_batch(self, batch):
        graphs_batch, node_indices_batch, edge_weights_batch, labels = batch

        labels = labels.to(self.device, non_blocking=self.non_blocking)
        graphs_batch = [g.to(self.device) for g in graphs_batch]
        node_indices_batch = [
            n.to(self.device, non_blocking=self.non_blocking) for n in node_indices_batch
        ]
        edge_weights_batch = [
            e.to(self.device, non_blocking=self.non_blocking) for e in edge_weights_batch
        ]
        return graphs_batch, node_indices_batch, edge_weights_batch, labels
        
    def train_single_graph_epoch(self, graph_idx, train_loader, optimizer):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in train_loader:
            g, node_indices, edge_weights, labels = self._move_single_graph_batch(
                batch, graph_idx
            )
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                node_features = self.model.embeddings(node_indices)
                final_features = self.model.graph_convolution(
                    g, node_features, edge_weights, graph_idx
                )
                doc_emb = self.model.document_embedding(g, final_features)
                logits = self.model.classify(doc_emb, graph_idx)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

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
                g, node_indices, edge_weights, labels = self._move_single_graph_batch(
                    batch, graph_idx
                )

                with torch.cuda.amp.autocast(enabled=self.use_amp):
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
        # Paper-Exact: Check if parallel training is enabled
        use_parallel = bool(self.config.get('use_parallel_stage1', True))
        
        if use_parallel and torch.cuda.device_count() > 1:
            # Paper-Exact: Use parallel training (2-4x speedup)
            self.train_stage1_parallel(train_loader, dev_loader)
        else:
            # Sequential training (original)
            self.train_stage1_sequential(train_loader, dev_loader)
    
    def train_stage1_sequential(self, train_loader, dev_loader):
        """Sequential training - fallback when parallel not available"""
        reuse_graph_states = bool(self.config.get('reuse_graph_states', False))

        for graph_idx in range(self.model.num_graphs):
            if reuse_graph_states and self.has_graph_state(graph_idx):
                print(f"\n{'='*60}")
                print(
                    f"Skipping Graph {graph_idx + 1}/{self.model.num_graphs} "
                    "because a saved checkpoint was found."
                )
                print(f"{'='*60}")
                self.load_graph_state(graph_idx)
                continue

            print(f"\n{'='*60}")
            print(f"Training Graph {graph_idx + 1}/{self.model.num_graphs}")
            print(f"{'='*60}")
            
            params = list(self.model.input_projections[graph_idx].parameters())
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
    
    def train_stage1_parallel(self, train_loader, dev_loader):
        """Paper-Exact: Parallel training on multiple GPUs (4x speedup)"""
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import tempfile
        
        print(f"\n{'='*60}")
        print("PARALLEL STAGE 1: Training 4 graphs simultaneously")
        print(f"{'='*60}\n")
        
        num_gpus = torch.cuda.device_count()
        print(f"Using {num_gpus} GPU(s) for parallel training\n")
        
        reuse_graph_states = bool(self.config.get('reuse_graph_states', False))
        temp_dir = tempfile.mkdtemp()
        
        # Check which graphs need training
        graphs_to_train = []
        for graph_idx in range(self.model.num_graphs):
            if reuse_graph_states and self.has_graph_state(graph_idx):
                print(f"Graph {graph_idx + 1}: Skip (checkpoint exists)")
                self.load_graph_state(graph_idx)
            else:
                graphs_to_train.append(graph_idx)
        
        # Train graphs in parallel
        def train_single_graph_worker(graph_idx):
            device = torch.device(f'cuda:{graph_idx % num_gpus}')
            model = self.model.to(device)
            model.train()
            
            params = list(model.input_projections[graph_idx].parameters())
            params += [model.edge_learning_params[graph_idx]]
            params += list(model.classifiers[graph_idx].parameters())
            
            optimizer = torch.optim.AdamW(params, lr=self.config['learning_rate'])
            best_val_loss = float('inf')
            patience_counter = 0
            patience = self.config['stage1_patience']
            
            for epoch in range(self.config['stage1_epochs']):
                train_loss, train_acc = self.train_single_graph_epoch(graph_idx, train_loader, optimizer)
                val_loss, val_acc = self.evaluate_single_graph(graph_idx, dev_loader)
                
                if epoch % 5 == 0:
                    print(f"Graph {graph_idx + 1} Epoch {epoch+1:3d}: "
                          f"Train Loss={train_loss:.4f} Train Acc={train_acc:.4f} | "
                          f"Val Loss={val_loss:.4f} Val Acc={val_acc:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save_graph_state(graph_idx)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Graph {graph_idx + 1}: Early stopping at epoch {epoch+1}")
                        break
            
            return graph_idx, best_val_loss
        
        # Execute parallel training
        with ProcessPoolExecutor(max_workers=min(len(graphs_to_train), num_gpus)) as executor:
            futures = {executor.submit(train_single_graph_worker, idx): idx 
                      for idx in graphs_to_train}
            
            for future in as_completed(futures):
                graph_idx, best_loss = future.result()
                print(f"✓ Graph {graph_idx + 1} complete (best val loss: {best_loss:.4f})")
        
        # Reload all states
        for graph_idx in range(self.model.num_graphs):
            self.load_graph_state(graph_idx)
        
        print(f"\n{'='*60}")
        print("PARALLEL STAGE 1 COMPLETE (2-4x speedup!)")
        print(f"{'='*60}\n")
    
    def train_stage2_epoch(self, train_loader, optimizer):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in train_loader:
            graphs_batch, node_indices_batch, edge_weights_batch, labels = self._move_full_batch(batch)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                predictions, fused_logits = self.model(
                    graphs_batch, node_indices_batch, edge_weights_batch
                )
                loss = self.criterion(fused_logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            total_loss += loss.item()
            preds = torch.argmax(fused_logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    def train_stage2(self, train_loader, dev_loader):
        stage2_mode = self.config.get('stage2_mode', 'joint')

        if stage2_mode == 'fusion_only':
            for param in self.model.input_projections.parameters():
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
        else:
            for param in self.model.parameters():
                param.requires_grad = True

            fusion_params = list(self.model.fusion_heads.parameters())
            fusion_param_ids = {id(param) for param in fusion_params}
            base_params = [
                param for param in self.model.parameters()
                if id(param) not in fusion_param_ids
            ]

            optimizer = torch.optim.AdamW(
                [
                    {'params': base_params, 'lr': self.config['learning_rate']},
                    {'params': fusion_params, 'lr': self.config['fusion_lr']},
                ]
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
                graphs_batch, node_indices_batch, edge_weights_batch, labels = self._move_full_batch(batch)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
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
        path = self.graph_state_path(graph_idx)
        state = {
            'input_projection': self.model.input_projections[graph_idx].state_dict(),
            'edge_learning': self.model.edge_learning_params[graph_idx].data.clone(),
            'classifier': self.model.classifiers[graph_idx].state_dict()
        }
        torch.save(state, path)
    
    def graph_state_path(self, graph_idx):
        return os.path.join(self.config['save_dir'], f'graph_{graph_idx}_best.pth')

    def has_graph_state(self, graph_idx):
        return os.path.exists(self.graph_state_path(graph_idx))

    def has_all_graph_states(self):
        return all(self.has_graph_state(graph_idx) for graph_idx in range(self.model.num_graphs))

    def graph_state_paths(self):
        return [self.graph_state_path(graph_idx) for graph_idx in range(self.model.num_graphs)]

    def load_graph_state(self, graph_idx, strict=True):
        path = self.graph_state_path(graph_idx)
        if not os.path.exists(path):
            if strict:
                raise FileNotFoundError(
                    f"Missing graph checkpoint for graph {graph_idx}: {path}"
                )
            return False

        state = torch.load(path, map_location=self.device)
        self.model.input_projections[graph_idx].load_state_dict(state['input_projection'])
        self.model.edge_learning_params[graph_idx].data.copy_(state['edge_learning'])
        self.model.classifiers[graph_idx].load_state_dict(state['classifier'])
        return True

    def load_all_graph_states(self, strict=True):
        loaded_all = True
        for graph_idx in range(self.model.num_graphs):
            loaded = self.load_graph_state(graph_idx, strict=strict)
            loaded_all = loaded_all and loaded
        return loaded_all

def run_training_pipeline(config):
    from gfn_custom_dataset import (
        CustomDatasetLoader,
        load_pretrained_embeddings,
        create_data_loaders,
    )
    from graph_construction import GraphConstructor
    from gfn_model import GraphFusionNetwork

    device = resolve_device(config.get('device', 'auto'))
    validate_graph_device(device)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    config = dict(config)
    config['device'] = str(device)
    config['data_dir'] = resolve_data_dir(config['data_dir'])
    config['embedding_path'] = resolve_embedding_path(config)
    loader_config = infer_loader_config(config, device)
    config.update(loader_config)
    os.makedirs(config['save_dir'], exist_ok=True)

    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(device.index or 0)
        print(f"Using device: {device} ({gpu_name})")
    else:
        print(f"Using device: {device}")
    print(f"Data directory: {config['data_dir']}")
    if config['embedding_path']:
        print(f"Embedding path: {config['embedding_path']}")
    elif config.get('embedding_preset', 'none') != 'none':
        print(f"Embedding preset: {config['embedding_preset']} (random fallback)")
    print(
        "DataLoader config: "
        f"num_workers={config['num_workers']} "
        f"pin_memory={config['pin_memory']} "
        f"persistent_workers={config['persistent_workers']}"
    )
    if config.get('amp') and device.type == 'cuda':
        print("Mixed precision: enabled")
    
    loader = CustomDatasetLoader(data_dir=config['data_dir'], tokenizer_mode=config.get('tokenizer_mode', 'auto'))
    (train_docs, train_labels, dev_docs, dev_labels, test_docs, test_labels, num_classes) = loader.load_all_splits(
        task=config['task'],
        remove_stopwords=config['remove_stopwords'],
        normalize_tone=config['normalize_tone'],
    )
    
    all_docs = train_docs + dev_docs + test_docs
    graph_docs = train_docs if config.get('graph_corpus_scope', 'train') == 'train' else all_docs

    constructor = GraphConstructor(
        window_size=config['window_size'],
        sequential_window=config['sequential_window'],
        sequential_edge_weight=config['sequential_edge_weight'],
        preserve_multiplicity=config['preserve_multiplicity'],
    )
    vocab = constructor.build_vocabulary(graph_docs, min_freq=config['min_freq'])
    embeddings = load_pretrained_embeddings(
        vocab, 
        embedding_dim=config['embedding_dim'],
        embedding_path=config.get('embedding_path')
    )
    corpus_graphs = constructor.build_all_graphs(
        graph_docs,
        embeddings,
        filter_edges=config.get('filter_edges', False),
    )

    train_loader, dev_loader, test_loader = create_data_loaders(
        train_docs, train_labels, dev_docs, dev_labels,
        test_docs, test_labels, constructor, corpus_graphs,
        batch_size=config['batch_size'],
        p_neighbors=config.get('p_neighbors'),
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        persistent_workers=config['persistent_workers'],
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
    
    trainer = GFNTrainer(model, device, config)

    pipeline_mode = config.get('pipeline_mode', 'full')

    if pipeline_mode == 'stage1_only':
        trainer.train_stage1(train_loader, dev_loader)
        results = {
            'task': config['task'],
            'status': 'stage1_complete',
            'graph_checkpoints': trainer.graph_state_paths(),
            'config': config,
        }
        results_path = os.path.join(config['save_dir'], 'stage1_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nStage 1 complete. Graph checkpoints saved for reuse.")
        print(f"Stage 1 summary saved to: {results_path}")
        return results

    if pipeline_mode == 'stage2_only':
        print("Loading saved graph checkpoints and skipping Stage 1.")
        trainer.load_all_graph_states(strict=True)
    else:
        trainer.train_stage1(train_loader, dev_loader)

    try:
        trainer.train_stage2(train_loader, dev_loader)
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user (Ctrl+C)")
        print("   Evaluating best model on test set...\n")
    
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
    parser.add_argument('--window_size', type=int, default=10)
    parser.add_argument('--min_freq', type=int, default=1)
    parser.add_argument('--graph_corpus_scope', type=str, default='all', choices=['train', 'all'])
    parser.add_argument('--sequential_window', type=int, default=1)
    parser.add_argument('--sequential_edge_weight', type=float, default=0.25)
    parser.add_argument('--embedding_dim', type=int, default=300)
    parser.add_argument('--hidden_dim', type=int, default=200)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--p_neighbors', type=int, default=5)
    parser.add_argument('--filter_edges', dest='filter_edges', action='store_true')
    parser.add_argument('--no_filter_edges', dest='filter_edges', action='store_false')
    parser.add_argument('--learning_rate', type=float, default=0.002)
    parser.add_argument('--fusion_lr', type=float, default=0.01)
    parser.add_argument('--stage2_mode', type=str, default='joint', choices=['joint', 'fusion_only'])
    parser.add_argument('--pipeline_mode', type=str, default='full', choices=['full', 'stage1_only', 'stage2_only'])
    parser.add_argument('--stage1_epochs', type=int, default=100)
    parser.add_argument('--stage1_patience', type=int, default=15)
    parser.add_argument('--stage2_iterations', type=int, default=1000)
    parser.add_argument('--stage2_patience', type=int, default=100)
    parser.add_argument('--reuse_graph_states', action='store_true')
    parser.add_argument('--remove_stopwords', action='store_true')
    parser.add_argument('--preserve_multiplicity', dest='preserve_multiplicity', action='store_true')
    parser.add_argument('--deduplicate_doc_tokens', dest='preserve_multiplicity', action='store_false')
    parser.add_argument('--normalize_tone', dest='normalize_tone', action='store_true')
    parser.add_argument('--no_normalize_tone', dest='normalize_tone', action='store_false')
    parser.add_argument(
        '--tokenizer_mode',
        type=str,
        default='auto',
        choices=['auto', 'whitespace', 'pretokenized'],
    )
    parser.add_argument('--embedding_path', type=str, default=None)
    parser.add_argument(
        '--embedding_preset',
        type=str,
        default='none',
        choices=list(EMBEDDING_PRESETS.keys()),
    )
    parser.add_argument('--glove_path', type=str, default=None)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--pin_memory', dest='pin_memory', action='store_true')
    parser.add_argument('--no_pin_memory', dest='pin_memory', action='store_false')
    parser.add_argument('--persistent_workers', dest='persistent_workers', action='store_true')
    parser.add_argument('--no_persistent_workers', dest='persistent_workers', action='store_false')
    parser.add_argument('--amp', dest='amp', action='store_true')
    parser.add_argument('--no_amp', dest='amp', action='store_false')
    parser.set_defaults(
        pin_memory=None,
        persistent_workers=None,
        amp=None,
        filter_edges=False,
        normalize_tone=True,
        preserve_multiplicity=True,
    )
    
    args = parser.parse_args()
    config = vars(args)

    if config['amp'] is None:
        config['amp'] = False
    
    results = run_training_pipeline(config)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)

if __name__ == "__main__":
    main()
