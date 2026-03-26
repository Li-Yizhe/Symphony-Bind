import argparse
import json
import os
import warnings
from typing import Dict, Any
from datetime import datetime
from time import strftime, localtime

def parse_args() -> Dict[str, Any]:
    """Parse and validate command line arguments."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Validate and process arguments
    validate_args(args)
    process_dataset_config(args)
    setup_output_dirs(args)
    setup_wandb_config(args)
    
    return args

def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser with all training arguments."""
    parser = argparse.ArgumentParser()
    
    # Model parameters
    add_model_args(parser)
    
    # Dataset parameters
    add_dataset_args(parser)
    
    # Training parameters
    add_training_args(parser)
    
    # Output parameters
    add_output_args(parser)
    
    # Wandb parameters
    add_wandb_args(parser)
    
    return parser

def add_model_args(parser: argparse.ArgumentParser):
    """Add model-related arguments."""
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--hidden_size', type=int, default=None)
    model_group.add_argument('--num_attention_head', type=int, default=8)
    model_group.add_argument('--attention_probs_dropout', type=float, default=0.1)
    model_group.add_argument('--plm_model', type=str, default='facebook/esm2_t33_650M_UR50D')
    model_group.add_argument('--pooling_method', type=str, default='convbert',
                            choices=['convbert', 'convbert_attention', 'convbert_max', 'mlp', 'bilstm', 'bilstm_pooling', 'cnn', 'transformer'])
    model_group.add_argument('--pooling_dropout', type=float, default=0.1)
    model_group.add_argument('--mlp_hidden_dims', type=int, nargs='+', default=[512, 256], 
                            help='Hidden dimensions for MLP layers')
    model_group.add_argument('--mlp_activation', type=str, default='relu', choices=['relu', 'gelu'],
                            help='Activation function for MLP layers')
    model_group.add_argument('--lstm_hidden_size', type=int, default=256,
                            help='Hidden size for LSTM layers')
    model_group.add_argument('--lstm_num_layers', type=int, default=2,
                            help='Number of LSTM layers')
    model_group.add_argument('--lstm_bidirectional', action='store_true', default=True,
                            help='Use bidirectional LSTM')
    model_group.add_argument('--bilstm_pooling', type=str, default='mean', choices=['mean', 'max', 'attention'],
                            help='Pooling method for BiLSTM pooling head')
    model_group.add_argument('--num_cnn_layers', type=int, default=3,
                            help='Number of CNN layers for CNN classification head')
    model_group.add_argument('--kernel_size', type=int, default=7,
                            help='Kernel size for CNN layers')
    model_group.add_argument('--nhead', type=int, default=8,
                            help='Number of attention heads for Transformer classification head')
    model_group.add_argument('--hidden_dim', type=int, default=512,
                            help='Hidden dimension for feedforward network in Transformer')
    model_group.add_argument('--feature_extractor_path', type=str, default=None,
                            help='Path to pre-trained feature extractor model (without decoder) to use as frozen feature extractor')
    model_group.add_argument('--plm_encoder_path', type=str, default=None,
                            help='Path to PLM-only model (created by remove_convbert.py) to use as frozen encoder')
    model_group.add_argument('--num_hidden_layers', type=int, default=1,
                            help='Number of Transformer encoder layers')

def add_dataset_args(parser: argparse.ArgumentParser):
    """Add dataset-related arguments."""
    data_group = parser.add_argument_group('Dataset Configuration')
    data_group.add_argument('--dataset', type=str)
    data_group.add_argument('--dataset_config', type=str)

    data_group.add_argument('--num_labels', type=int)
    data_group.add_argument('--problem_type', type=str)
    data_group.add_argument('--pdb_type', type=str)
    data_group.add_argument('--train_file', type=str)
    data_group.add_argument('--valid_file', type=str)
    data_group.add_argument('--test_file', type=str)
    data_group.add_argument('--metrics', nargs='+', type=str)
    data_group.add_argument('--data_group_dir', type=str, default='dataset/data_group',
                           help='Directory containing grouped data for MTL (e.g., dataset/data_group)')
    data_group.add_argument('--task_names', type=str, nargs='+', default=None,
                           help='List of task names (molecule types) for MTL, e.g., ADP ATP AMP')
    data_group.add_argument('--mtl_group', type=str, default=None,
                           help='Specific group name to train for MTL (e.g., nucleotide, inorganic_ion, cofactor). If not specified, trains all groups separately.')

def add_training_args(parser: argparse.ArgumentParser):
    """Add training-related arguments."""
    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility. Default is 42.')
    train_group.add_argument('--learning_rate', type=float, default=1e-3)
    train_group.add_argument('--scheduler', type=str, choices=['linear', 'cosine', 'step'])
    train_group.add_argument('--warmup_steps', type=int, default=0)
    train_group.add_argument('--num_workers', type=int, default=4)
    train_group.add_argument('--batch_size', type=int)
    train_group.add_argument('--batch_token', type=int)
    train_group.add_argument('--num_epochs', type=int, default=100)
    train_group.add_argument('--max_seq_len', type=int, default=-1)
    train_group.add_argument('--gradient_accumulation_steps', type=int, default=1)
    train_group.add_argument('--max_grad_norm', type=float, default=-1)
    train_group.add_argument('--patience', type=int, default=10)
    train_group.add_argument('--min_delta', type=float, default=0.0, help="Minimum change in monitored metric to qualify as an improvement")
    train_group.add_argument('--min_epochs', type=int, default=0, help="Minimum number of epochs before early stopping can be triggered")
    train_group.add_argument('--monitor', type=str)
    train_group.add_argument('--monitor_strategy', type=str, choices=['max', 'min'])
    train_group.add_argument('--loss_type', type=str, default='combined', 
                           choices=['focal', 'weighted_bce', 'combined'],
                           help='Type of loss function to use for ablation studies')
    train_group.add_argument('--training_method', type=str, default='freeze',
                           choices=['full', 'freeze', 'lora', 'ses-adapter', 'plm-lora', 'plm-qlora', 'plm-adalora', 'plm-dora', 'plm-ia3', 'mtl', 'mtl-lora'])
    parser.add_argument("--lora_r", type=int, default=8, help="lora r")
    parser.add_argument("--lora_alpha", type=int, default=32, help="lora_alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="lora_dropout")
    parser.add_argument("--feedforward_modules", type=str, default="w0")

    parser.add_argument(
        "--lora_target_modules",
        nargs="+",
        default=["query", "key", "value"],
        help="lora target module",
    )
    train_group.add_argument('--structure_seq', type=str, default='')
    train_group.add_argument('--test_only', action='store_true', help='Only run testing, skip training')

def add_output_args(parser: argparse.ArgumentParser):
    """Add output-related arguments."""
    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument('--output_model_name', type=str)
    output_group.add_argument('--output_root', default="ckpt")
    output_group.add_argument('--output_dir', default=None)

def add_wandb_args(parser: argparse.ArgumentParser):
    """Add wandb-related arguments."""
    wandb_group = parser.add_argument_group('Wandb Configuration')
    wandb_group.add_argument('--wandb', action='store_true')
    wandb_group.add_argument('--wandb_entity', type=str)
    wandb_group.add_argument('--wandb_project', type=str, default='VenusFactory')
    wandb_group.add_argument('--wandb_run_name', type=str)

def validate_args(args: argparse.Namespace):
    """Validate command line arguments."""
    if args.batch_size is None and args.batch_token is None:
        raise ValueError("batch_size or batch_token must be provided")
    
    if args.training_method == 'ses-adapter':
        if args.structure_seq is None:
            raise ValueError("structure_seq must be provided for ses-adapter")
        args.structure_seq = args.structure_seq.split(',')
    else:
        args.structure_seq = []

def process_dataset_config(args: argparse.Namespace):
    """Process dataset configuration file."""

    # Handle metrics specially
    if args.metrics:
        if isinstance(args.metrics, str):
            args.metrics = args.metrics.split(',')
        if args.metrics == ['None']:
            args.metrics = ['loss']
            warnings.warn("No metrics provided, using default metrics: loss")

    if not args.dataset_config:
        return
        
    # Check if config file exists and is not empty
    if not os.path.exists(args.dataset_config):
        print(f"Warning: Dataset config file {args.dataset_config} not found, skipping config processing")
        return
        
    try:
        with open(args.dataset_config, 'r') as f:
            config_content = f.read().strip()
            if not config_content:
                print(f"Warning: Dataset config file {args.dataset_config} is empty, skipping config processing")
                return
            config = json.loads(config_content)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in dataset config file {args.dataset_config}: {e}")
        return
    
    # Update args with dataset config values if not already set
    for key in ['dataset', 'pdb_type', 'train_file', 'valid_file', 'test_file',
                'num_labels', 'problem_type', 'monitor', 'monitor_strategy', 
                'metrics']:
        if getattr(args, key) is None and key in config:
            setattr(args, key, config[key])
    
    # Handle metrics specially
    if args.metrics:
        args.metrics = args.metrics.split(',')
        if args.metrics == ['None']:
            args.metrics = ['loss']
            warnings.warn("No metrics provided, using default metrics: loss")

def setup_output_dirs(args: argparse.Namespace):
    """Setup output directories."""
    if args.output_dir is None:
        current_date = strftime("%Y%m%d", localtime())
        args.output_dir = os.path.join(args.output_root, current_date)
    else:
        # If output_dir is already an absolute path or starts with ./, don't join with output_root
        if os.path.isabs(args.output_dir) or args.output_dir.startswith('./') or args.output_dir.startswith('../'):
            pass  # Keep the path as is
        else:
            args.output_dir = os.path.join(args.output_root, args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

def setup_wandb_config(args: argparse.Namespace):
    """Setup wandb configuration."""
    if args.wandb:
        if args.wandb_run_name is None:
            args.wandb_run_name = f"VenusFactory-{args.dataset}"
        if args.output_model_name is None:
            args.output_model_name = f"{args.wandb_run_name}.pt" 
