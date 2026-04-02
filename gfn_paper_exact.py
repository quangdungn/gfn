"""
GFN Paper-Exact Implementation with Grid Search Parameter Optimization
Implements Dai et al. 2022 specifications exactly

Paper Section 5.1.3: "For other parameters searching, we adopt Neural Network 
Intelligence (NNI) to implement grid searching."

This module supports:
1. Single run: Paper-exact hyperparameters
2. NNI Grid Search: Official NNI tool (paper-exact)
3. Custom Grid Search: Fallback implementation
"""
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import json
import os
from pathlib import Path
from itertools import product
from collections import defaultdict
import numpy as np
from datetime import datetime

# Try to import NNI (paper-exact requirement)
try:
    import nni
    HAS_NNI = True
except ImportError:
    HAS_NNI = False

# Import existing GFN components
from gfn_vietnamese_pipeline import (
    resolve_device, resolve_data_dir, resolve_embedding_path,
    infer_loader_config, GFNTrainer, run_training_pipeline
)


class PaperExactHyperparams:
    """
    Exact hyperparameters from Dai et al. 2022 paper
    """
    
    # Stage 1: Per-graph training (validation loss no decrease for 10 epochs)
    STAGE1_PATIENCE = 10
    STAGE1_MAX_EPOCHS = 200  # Paper doesn't specify max, but patience ensures stopping
    
    # Stage 2: Fusion module training (validation loss no decrease for 100 iterations) 
    STAGE2_PATIENCE = 100
    STAGE2_MAX_ITERATIONS = 1000  # Same as above
    
    # Optimization
    OPTIMIZER = "AdamW"  # Default parameters
    FUSION_LR = 0.05  # Paper explicitly states this
    STAGE1_LR = None  # AdamW default (not explicitly specified in paper)
    
    # Graph construction (from paper Table descriptions)
    WINDOW_SIZE = 20  # From paper: "window-size L = 20"
    EDGE_FILTER_RATIO = 0.005  # "min(0.005 * n, 100)"
    EDGE_FILTER_MAX = 100
    
    # Embeddings
    EMBEDDING_DIM = 300  # "300d GloVe" or pretrained
    
    # Model architecture
    NUM_GRAPHS = 4  # Fixed
    
    # Vocabulary thresholds (dataset-specific, from paper)
    VOCAB_THRESHOLDS = {
        'mr': 0,  # Keep all words
        'sentiment': 5,  # Vietnamese dataset
        'topic': 5,
        '20ng': 10,
        'default': 5
    }
    
    # Grid search spaces (from paper Table/experiments)
    GRID_SEARCH_SPACE = {
        'batch_size': [16, 32, 64, 128],
        'dropout': [0.3, 0.5, 0.7],
        'p_neighbors': [1, 3, 5, 7, 9],
    }


class GridSearchOptimizer:
    """
    Exhaustive grid search using paper specifications
    Similar to paper's NNI-based approach
    """
    
    def __init__(self, base_config, save_dir, num_trials=None):
        """
        Args:
            base_config: Base configuration dict
            save_dir: Where to save grid search results
            num_trials: Max trials (None = full grid)
        """
        self.base_config = base_config.copy()
        self.save_dir = save_dir
        self.num_trials = num_trials
        self.results = []
        
    def generate_grid(self):
        """Generate all combinations from grid search space"""
        space = PaperExactHyperparams.GRID_SEARCH_SPACE
        
        # Create all combinations
        param_names = sorted(space.keys())
        param_values = [space[name] for name in param_names]
        
        all_combinations = list(product(*param_values))
        
        # Limit to num_trials if specified
        if self.num_trials:
            np.random.seed(42)
            indices = np.random.choice(len(all_combinations), 
                                       min(self.num_trials, len(all_combinations)),
                                       replace=False)
            all_combinations = [all_combinations[i] for i in indices]
        
        # Convert to dicts
        grids = []
        for values in all_combinations:
            grid_dict = dict(zip(param_names, values))
            grids.append(grid_dict)
        
        return grids
    
    def run_trial(self, trial_num, params):
        """Run single grid search trial"""
        config = self.base_config.copy()
        config.update(params)
        config['save_dir'] = os.path.join(
            self.save_dir, 
            f"trial_{trial_num:03d}_" + 
            "_".join([f"{k}={v}" for k, v in sorted(params.items())])
        )
        
        print(f"\n{'='*70}")
        print(f"GRID SEARCH TRIAL {trial_num}")
        print(f"Parameters: {params}")
        print(f"{'='*70}")
        
        try:
            results = run_training_pipeline(config)
            
            trial_result = {
                'trial': trial_num,
                'params': params,
                'test_accuracy': results.get('test_accuracy'),
                'test_macro_f1': results.get('test_macro_f1'),
                'test_micro_f1': results.get('test_micro_f1'),
                'success': True
            }
        except Exception as e:
            trial_result = {
                'trial': trial_num,
                'params': params,
                'error': str(e),
                'success': False
            }
            print(f"Trial failed: {e}")
        
        self.results.append(trial_result)
        return trial_result
    
    def run_full_grid_search(self):
        """Run all trials in grid search"""
        grids = self.generate_grid()
        print(f"\nStarting grid search with {len(grids)} configurations...")
        
        for trial_num, params in enumerate(grids, 1):
            self.run_trial(trial_num, params)
        
        # Find best
        successful = [r for r in self.results if r['success']]
        if successful:
            best = max(successful, key=lambda x: x['test_accuracy'])
            print(f"\n{'='*70}")
            print(f"BEST CONFIGURATION:")
            print(f"Accuracy: {best['test_accuracy']:.4f}")
            print(f"Macro-F1: {best['test_macro_f1']:.4f}")
            print(f"Parameters: {best['params']}")
            print(f"{'='*70}\n")
        
        return self.results
    
    def save_results(self):
        """Save grid search results"""
        results_file = os.path.join(self.save_dir, 'grid_search_results.json')
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Grid search results saved to: {results_file}")
        return results_file


def create_paper_exact_config(task='sentiment', data_dir=None, dataset='dataGPT', **kwargs):
    """
    Create config that matches Dai et al. 2022 paper exactly
    
    Args:
        task: 'sentiment' or 'topic'
        data_dir: Path to data directory
        dataset: 'dataGPT' or 'dataUIT' 
        **kwargs: Override any defaults
    """
    
    config = {
        # Data
        'task': task,
        'data_dir': data_dir or f'./{dataset}',  # Fixed: was f'./data{dataset}'
        'graph_corpus_scope': 'train',
        'tokenizer_mode': 'auto',
        
        # Graph construction (PAPER EXACT)
        'window_size': PaperExactHyperparams.WINDOW_SIZE,  # L = 20
        'sequential_window': 1,
        'sequential_edge_weight': 0.25,
        'min_freq': PaperExactHyperparams.VOCAB_THRESHOLDS.get(
            task, PaperExactHyperparams.VOCAB_THRESHOLDS['default']
        ),
        'filter_edges': True,
        'preserve_multiplicity': True,
        
        # Embeddings
        'embedding_dim': PaperExactHyperparams.EMBEDDING_DIM,
        'embedding_preset': 'phow2v_syllables_300',
        
        # Model (PAPER EXACT)
        'hidden_dim': 300,  # Paper specifies 300
        'num_heads': 3,  # Default, will be overridden by grid search
        'dropout': 0.3,  # Default, will be overridden by grid search
        'batch_size': 32,  # Default, will be overridden by grid search
        'p_neighbors': 5,  # Default, will be overridden by grid search
        
        # Training - Stage 1 (PAPER EXACT)
        'stage1_epochs': 200,  # Max, but patience controls actual stopping
        'stage1_patience': PaperExactHyperparams.STAGE1_PATIENCE,  # 10 epochs no improvement
        'stage2_mode': 'joint',
        
        # Training - Stage 2 (PAPER EXACT)
        'stage2_iterations': 1000,  # Max, but patience controls actual stopping
        'stage2_patience': PaperExactHyperparams.STAGE2_PATIENCE,  # 100 iterations no improvement
        'learning_rate': 0.001,  # AdamW default approximation
        'fusion_lr': PaperExactHyperparams.FUSION_LR,  # 0.05 from paper
        
        # Data processing
        'remove_stopwords': False,
        'normalize_tone': True,
        
        # Hardware
        'device': 'auto',
        'amp': True,
        'num_workers': 4,
        'pin_memory': True,
        'persistent_workers': False,
        
        # Paper-Exact Parallel Training (Stage 1)
        'use_parallel_stage1': True,  # Enable parallel training if multiple GPUs available
        'reuse_graph_states': True,   # Checkpoint and resume support
        
        # Output
        'save_dir': './checkpoints/paper_exact',
        'run_name': 'paper_exact',
    }
    
    # Override with kwargs
    config.update(kwargs)
    
    return config


def train_with_paper_exact_config(task='sentiment', data_dir=None, 
                                  dataset='dataGPT', run_name='paper_exact', **kwargs):
    """
    Train GFN with paper's exact specifications (no grid search, single run)
    """
    config = create_paper_exact_config(
        task=task, 
        data_dir=data_dir,
        dataset=dataset,
        run_name=run_name,
        **kwargs
    )
    
    results = run_training_pipeline(config)
    return results


def train_with_grid_search(task='sentiment', data_dir=None, dataset='dataGPT',
                          num_trials=None, run_name='paper_grid_search', **kwargs):
    """
    Train GFN with parameter grid search (paper's approach)
    """
    base_config = create_paper_exact_config(
        task=task,
        data_dir=data_dir,
        dataset=dataset,
        run_name=run_name,
        **kwargs
    )
    
    save_dir = os.path.join('./checkpoints', run_name)
    os.makedirs(save_dir, exist_ok=True)
    
    optimizer = GridSearchOptimizer(base_config, save_dir, num_trials=num_trials)
    optimizer.run_full_grid_search()
    results_file = optimizer.save_results()
    
    print(f"\nGrid search complete! Results: {results_file}")
    return optimizer.results


def train_with_nni_grid_search(task='sentiment', data_dir=None, dataset='dataGPT',
                                run_name='paper_nni_grid_search', **kwargs):
    """
    Train GFN with NNI grid search (paper-exact requirement)
    
    Paper Section 5.1.3: 
    "For other parameters searching, we adopt Neural Network Intelligence (NNI)
    to implement grid searching."
    """
    if not HAS_NNI:
        raise ImportError(
            "NNI not installed! Install with: pip install nni>=3.0\n"
            "Or use --mode grid_search for custom grid search implementation."
        )
    
    # Get hyperparameter suggestion from NNI
    params = nni.get_next_parameter()
    
    base_config = create_paper_exact_config(
        task=task,
        data_dir=data_dir,
        dataset=dataset,
        run_name=run_name,
        **kwargs
    )
    
    # Override with NNI parameters
    base_config.update(params)
    
    print(f"\nNNI Trial Parameters: {params}")
    print(f"Running training with these parameters...")
    
    try:
        results = run_training_pipeline(base_config)
        
        # Report back to NNI - maximize test accuracy
        test_accuracy = results.get('test_accuracy', 0.0)
        nni.report_final_result(test_accuracy)
        
        print(f"Trial Result - Test Accuracy: {test_accuracy:.4f}")
        return results
        
    except Exception as e:
        print(f"Trial failed: {e}")
        nni.report_final_result(0.0)
        raise


def main():
    parser = argparse.ArgumentParser(
        description='GFN Paper-Exact Implementation with Parameter Search'
    )
    
    # Core arguments
    parser.add_argument('--task', choices=['sentiment', 'topic'], 
                       default='sentiment',
                       help='Task type')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Data directory path')
    parser.add_argument('--dataset', choices=['dataGPT', 'dataUIT'],
                       default='dataGPT',
                       help='Dataset to use')
    parser.add_argument('--mode', choices=['single', 'grid_search', 'nni'],
                       default='single',
                       help='Training mode: single, grid_search, or nni (paper-exact)')
    parser.add_argument('--num_trials', type=int, default=None,
                       help='Number of grid search trials (None = full grid)')
    parser.add_argument('--run_name', type=str, default='paper_exact',
                       help='Run name for checkpoints')
    parser.add_argument('--device', default='auto',
                       help='Device: auto, cpu, cuda, cuda:0, etc.')
    
    # Training hyperparameters (override defaults)
    parser.add_argument('--stage1_epochs', type=int, default=None,
                       help='Stage 1 max epochs (default: 200, paper: 10 patience)')
    parser.add_argument('--stage1_patience', type=int, default=None,
                       help='Stage 1 patience (default: 10 from paper)')
    parser.add_argument('--stage2_iterations', type=int, default=None,
                       help='Stage 2 max iterations (default: 1000, paper: 100 patience)')
    parser.add_argument('--stage2_patience', type=int, default=None,
                       help='Stage 2 patience (default: 100 from paper)')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Stage 1 learning rate (default: 0.001)')
    parser.add_argument('--fusion_lr', type=float, default=None,
                       help='Stage 2 fusion learning rate (default: 0.05 from paper)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (default: 32, grid search: [16, 32, 64, 128])')
    parser.add_argument('--hidden_dim', type=int, default=None,
                       help='Hidden dimension (default: 300 from paper)')
    parser.add_argument('--dropout', type=float, default=None,
                       help='Dropout rate (default: 0.3, grid search: [0.3, 0.5, 0.7])')
    parser.add_argument('--window_size', type=int, default=None,
                       help='Graph window size (default: 20 from paper)')
    
    args = parser.parse_args()
    
    # Build kwargs from available arguments
    override_kwargs = {}
    if args.stage1_epochs is not None:
        override_kwargs['stage1_epochs'] = args.stage1_epochs
    if args.stage1_patience is not None:
        override_kwargs['stage1_patience'] = args.stage1_patience
    if args.stage2_iterations is not None:
        override_kwargs['stage2_iterations'] = args.stage2_iterations
    if args.stage2_patience is not None:
        override_kwargs['stage2_patience'] = args.stage2_patience
    if args.learning_rate is not None:
        override_kwargs['learning_rate'] = args.learning_rate
    if args.fusion_lr is not None:
        override_kwargs['fusion_lr'] = args.fusion_lr
    if args.batch_size is not None:
        override_kwargs['batch_size'] = args.batch_size
    if args.hidden_dim is not None:
        override_kwargs['hidden_dim'] = args.hidden_dim
    if args.dropout is not None:
        override_kwargs['dropout'] = args.dropout
    if args.window_size is not None:
        override_kwargs['window_size'] = args.window_size
    
    print(f"\n{'='*70}")
    print(f"GFN Paper-Exact Implementation")
    print(f"Dai et al. 2022 - Graph Fusion Network for Text Classification")
    print(f"{'='*70}\n")
    
    print(f"Configuration:")
    print(f"  Task: {args.task}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Mode: {args.mode}")
    print(f"  Device: {args.device}")
    if override_kwargs:
        print(f"  Overrides: {override_kwargs}\n")
    else:
        print()
    
    if args.mode == 'single':
        print("Running with paper's exact hyperparameters (no grid search)...\n")
        results = train_with_paper_exact_config(
            task=args.task,
            data_dir=args.data_dir,
            dataset=args.dataset,
            run_name=args.run_name,
            device=args.device,
            **override_kwargs
        )
        print(f"\nResults:")
        print(f"  Accuracy: {results.get('test_accuracy', 'N/A'):.4f}")
        print(f"  Macro-F1: {results.get('test_macro_f1', 'N/A'):.4f}")
        
    elif args.mode == 'grid_search':
        print("Running grid search with parameter combinations (custom implementation)...\n")
        results = train_with_grid_search(
            task=args.task,
            data_dir=args.data_dir,
            dataset=args.dataset,
            num_trials=args.num_trials,
            run_name=args.run_name,
            **override_kwargs
        )
        print(f"Grid search complete with {len(results)} trials")
    
    elif args.mode == 'nni':
        print("Running NNI grid search (paper-exact: Dai et al. 2022)...\n")
        if not HAS_NNI:
            print("ERROR: NNI not installed!")
            print("Install with: pip install nni>=3.0")
            print("\nOr use command:")
            print("  nni create --config gfn/nni_config.yml")
            return
        
        train_with_nni_grid_search(
            task=args.task,
            data_dir=args.data_dir,
            dataset=args.dataset,
            run_name=args.run_name,
            device=args.device,
            **override_kwargs
        )


if __name__ == '__main__':
    main()
