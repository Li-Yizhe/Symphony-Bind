import os
import sys
import json
import torch
import random
import numpy as np

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from utils.args import parse_args
from utils.logger import setup_logging, print_model_parameters
from dataset.dataloader.dataloader import prepare_dataloaders
from models.finetune_models.peft_model_factory import create_models, peft_factory

# Import trainer using direct path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from trainer import Trainer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # Parse arguments
    args = parse_args()
    
    # Use default seed 42 if not explicitly set
    if args.seed is None:
        args.seed = 42
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Setup logging and wandb
    logger = setup_logging(args)
    logger.info(f"Random seed set to: {args.seed}")
    
    if args.wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            config=vars(args)
        )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize models and tokenizer
    logger.info("=" * 60)
    logger.info("Initializing models and tokenizer...")
    if args.training_method in ['plm-lora', 'plm-qlora', 'plm-dora', 'plm-adalora', 'plm-ia3']:
        model, plm_model, tokenizer = peft_factory(args)
    else:
        model, plm_model, tokenizer = create_models(args)
    print_model_parameters(model, plm_model, logger)
    logger.info("=" * 60)
    
    # Prepare data loaders
    train_loader, val_loader, test_loader = prepare_dataloaders(args, tokenizer, logger)
    
    # Create trainer
    trainer = Trainer(args, model, plm_model, logger, train_loader)
    
    # Check if we're only testing
    if hasattr(args, 'test_only') and args.test_only:
        # Only run testing
        if test_loader is not None:
            trainer.test(test_loader, tokenizer)
        else:
            logger.error("Test loader is None. Cannot run testing.")
    else:
        # Train and validate (no testing)
        trainer.train(train_loader, val_loader)
    
    if args.wandb:
        import wandb
        wandb.finish()

if __name__ == "__main__":
    main()