import os
import json
import wandb
from utils.args import parse_args
from utils.logger import setup_logging, print_model_parameters
from dataset.dataloader.dataloader import prepare_dataloaders
from models.finetune_models.peft_model_factory import create_models, peft_factory
from .trainer import Trainer

def main():
    # Parse arguments
    args = parse_args()
    
    # Setup logging and wandb
    logger = setup_logging(args)
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            config=vars(args)
        )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize models and tokenizer
    if args.training_method in ['plm-lora', 'plm-qlora', 'plm-dora', 'plm-adalora', 'plm-ia3']:
        model, plm_model, tokenizer = peft_factory(args)
    else:
        model, plm_model, tokenizer = create_models(args)
    print_model_parameters(model, plm_model, logger)
    
    # Prepare data with tokenizer
    train_loader, val_loader, test_loader = prepare_dataloaders(args, tokenizer, logger)
    
    # Create trainer
    trainer = Trainer(args, model, plm_model, logger, train_loader)
    
    # Train and validate (no testing)
    trainer.train(train_loader, val_loader)
    
    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()