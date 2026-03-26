import json
import torch
import random
import numpy as np
import datasets
from torch.utils.data import DataLoader
from dataset.dataloader.collator import Collator
from dataset.dataloader.batch_sampler import BatchSampler
from torch.utils.data import Dataset
from typing import Dict, Any, List, Union
import pandas as pd

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def prepare_dataloaders(args, tokenizer, logger):
    """Prepare train, validation and test dataloaders."""
    # Check if MTL training
    if args.training_method in ['mtl', 'mtl-lora']:
        return prepare_mtl_dataloaders(args, tokenizer, logger)
    
    if args.dataset == "csv":
        # Load CSV files directly
        train_dataset = pd.read_csv(args.train_file)
        val_dataset = pd.read_csv(args.valid_file)
        
        # Handle test file if provided
        if args.test_file:
            test_dataset = pd.read_csv(args.test_file)
        else:
            # Create empty test dataset if not provided
            test_dataset = pd.DataFrame(columns=train_dataset.columns)
        
        # Convert to list of dictionaries format
        train_dataset = train_dataset.to_dict('records')
        val_dataset = val_dataset.to_dict('records')
        test_dataset = test_dataset.to_dict('records')
    else:
        # Use HuggingFace datasets
        train_dataset = datasets.load_dataset(args.dataset)['train']
        val_dataset = datasets.load_dataset(args.dataset)['validation']
        test_dataset = datasets.load_dataset(args.dataset)['test']
    
    train_dataset_token_lengths = [len(item['sequence']) for item in train_dataset]
    val_dataset_token_lengths = [len(item['sequence']) for item in val_dataset]
    test_dataset_token_lengths = [len(item['sequence']) for item in test_dataset]
    
    # log dataset info
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"  Number of train samples: {len(train_dataset)}")
    logger.info(f"  Number of val samples: {len(val_dataset)}")
    logger.info(f"  Number of test samples: {len(test_dataset)}")
    
    # log 3 data points from train_dataset
    logger.info("Sample 3 data points from train dataset:")
    logger.info(f"  Train data point 1: {train_dataset[0]}")
    logger.info(f"  Train data point 2: {train_dataset[1]}")
    logger.info(f"  Train data point 3: {train_dataset[2]}")
    
    collator = Collator(
        tokenizer=tokenizer,
        max_length=args.max_seq_len if args.max_seq_len > 0 else None,
        problem_type=args.problem_type,
        num_labels=args.num_labels
    )
    
    # Create generator for reproducibility
    g = torch.Generator()
    g.manual_seed(args.seed)
    
    # Common dataloader parameters
    dataloader_params = {
        'num_workers': args.num_workers,
        'collate_fn': collator,
        'pin_memory': True,
        'persistent_workers': True if args.num_workers > 0 else False,
        'prefetch_factor': 2,
        'worker_init_fn': worker_init_fn,
        'generator': g,
    }
    
    # Create dataloaders based on batching strategy
    if args.batch_token is not None:
        train_loader = create_token_based_loader(train_dataset, train_dataset_token_lengths, args.batch_token, True, **dataloader_params)
        val_loader = create_token_based_loader(val_dataset, val_dataset_token_lengths, args.batch_token, False, **dataloader_params)
        test_loader = create_token_based_loader(test_dataset, test_dataset_token_lengths, args.batch_token, False, **dataloader_params)
    else:
        train_loader = create_size_based_loader(train_dataset, args.batch_size, True, **dataloader_params)
        val_loader = create_size_based_loader(val_dataset, args.batch_size, False, **dataloader_params)
        test_loader = create_size_based_loader(test_dataset, args.batch_size, False, **dataloader_params)
    
    return train_loader, val_loader, test_loader

def create_token_based_loader(dataset, token_lengths, batch_token, shuffle, **kwargs):
    """Create dataloader with token-based batching."""
    sampler = BatchSampler(token_lengths, batch_token, shuffle=shuffle)
    return DataLoader(dataset, batch_sampler=sampler, **kwargs)

def create_size_based_loader(dataset, batch_size, shuffle, **kwargs):
    """Create dataloader with size-based batching."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)


def prepare_mtl_dataloaders(args, tokenizer, logger):
    """
    Prepare MTL dataloaders from data_group directory structure.
    
    Structure: data_group/{category}/{molecule}/{train,val,test}.csv
    Each molecule type is a task. Tasks within the same category (group) share an encoder.
    If mtl_group is specified, only loads data for that specific group.
    """
    import os
    from models.finetune_models.peft_model_factory import get_mtl_group_to_tasks
    
    # Get group-to-tasks mapping (only for specified group if mtl_group is set)
    group_to_tasks = get_mtl_group_to_tasks(args)
    args.group_to_tasks = group_to_tasks
    
    # If mtl_group is specified, log which group is being trained
    mtl_group = getattr(args, 'mtl_group', None)
    if mtl_group:
        logger.info(f"Training MTL model for group: {mtl_group}")
    
    # Create flat task names list and mappings
    task_names = []
    task_name_to_group = {}
    group_name_to_idx = {}
    task_idx = 0
    
    for group_idx, (group_name, tasks) in enumerate(group_to_tasks.items()):
        group_name_to_idx[group_name] = group_idx
        for task_name in tasks:
            task_names.append(task_name)
            task_name_to_group[task_name] = group_name
            task_idx += 1
    
    args.task_names = task_names
    args.task_name_to_group = task_name_to_group
    args.group_name_to_idx = group_name_to_idx
    
    logger.info(f"MTL Groups and Tasks:")
    for group_name, tasks in group_to_tasks.items():
        logger.info(f"  Group {group_name}: {tasks}")
    
    data_group_dir = getattr(args, 'data_group_dir', 'dataset/data_group')
    
    # Load data for each task
    train_datasets = []
    val_datasets = []
    test_datasets = []
    
    for task_idx, task_name in enumerate(task_names):
        # Get group name for this task
        group_name = task_name_to_group[task_name]
        group_idx = group_name_to_idx[group_name]
        
        # Find task data in data_group directory
        task_path = os.path.join(data_group_dir, group_name, task_name)
        if os.path.isdir(task_path):
            train_file = os.path.join(task_path, 'train.csv')
            val_file = os.path.join(task_path, 'val.csv')
            test_file = os.path.join(task_path, 'test.csv')
            
            if os.path.exists(train_file):
                # Load data
                train_df = pd.read_csv(train_file)
                val_df = pd.read_csv(val_file) if os.path.exists(val_file) else pd.DataFrame()
                test_df = pd.read_csv(test_file) if os.path.exists(test_file) else pd.DataFrame()
                
                # Add task_id, group_id, task_name, and group_name to each sample
                train_df['task_id'] = task_idx
                train_df['group_id'] = group_idx
                train_df['task_name'] = task_name
                train_df['group_name'] = group_name
                val_df['task_id'] = task_idx
                val_df['group_id'] = group_idx
                val_df['task_name'] = task_name
                val_df['group_name'] = group_name
                test_df['task_id'] = task_idx
                test_df['group_id'] = group_idx
                test_df['task_name'] = task_name
                test_df['group_name'] = group_name
                
                train_datasets.append(train_df)
                val_datasets.append(val_df)
                test_datasets.append(test_df)
                
                logger.info(f"  Task {task_name} (Group {group_name}): {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
            else:
                logger.warning(f"Task {task_name} train file not found: {train_file}")
        else:
            logger.warning(f"Task {task_name} directory not found: {task_path}")
    
    # Combine all tasks
    if train_datasets:
        train_dataset = pd.concat(train_datasets, ignore_index=True)
        val_dataset = pd.concat(val_datasets, ignore_index=True) if val_datasets else pd.DataFrame()
        test_dataset = pd.concat(test_datasets, ignore_index=True) if test_datasets else pd.DataFrame()
    else:
        raise ValueError(f"No task data found in {data_group_dir}")
    
    # Shuffle training data
    train_dataset = train_dataset.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    
    # Convert to list of dictionaries
    train_dataset = train_dataset.to_dict('records')
    val_dataset = val_dataset.to_dict('records')
    test_dataset = test_dataset.to_dict('records')
    
    logger.info(f"MTL Dataset Summary:")
    logger.info(f"  Total train samples: {len(train_dataset)}")
    logger.info(f"  Total val samples: {len(val_dataset)}")
    logger.info(f"  Total test samples: {len(test_dataset)}")
    
    train_dataset_token_lengths = [len(item['sequence']) for item in train_dataset]
    val_dataset_token_lengths = [len(item['sequence']) for item in val_dataset]
    test_dataset_token_lengths = [len(item['sequence']) for item in test_dataset]
    
    # Create MTL collator that handles task_id
    from dataset.dataloader.collator import MTLCollator
    collator = MTLCollator(
        tokenizer=tokenizer,
        max_length=args.max_seq_len if args.max_seq_len > 0 else None,
        problem_type=args.problem_type,
        num_labels=args.num_labels
    )
    
    # Create generator for reproducibility
    g = torch.Generator()
    g.manual_seed(args.seed)
    
    # Common dataloader parameters
    dataloader_params = {
        'num_workers': args.num_workers,
        'collate_fn': collator,
        'pin_memory': True,
        'persistent_workers': True if args.num_workers > 0 else False,
        'prefetch_factor': 2,
        'worker_init_fn': worker_init_fn,
        'generator': g,
    }
    
    # Create dataloaders based on batching strategy
    if args.batch_token is not None:
        train_loader = create_token_based_loader(train_dataset, train_dataset_token_lengths, args.batch_token, True, **dataloader_params)
        val_loader = create_token_based_loader(val_dataset, val_dataset_token_lengths, args.batch_token, False, **dataloader_params)
        test_loader = create_token_based_loader(test_dataset, test_dataset_token_lengths, args.batch_token, False, **dataloader_params)
    else:
        train_loader = create_size_based_loader(train_dataset, args.batch_size, True, **dataloader_params)
        val_loader = create_size_based_loader(val_dataset, args.batch_size, False, **dataloader_params)
        test_loader = create_size_based_loader(test_dataset, args.batch_size, False, **dataloader_params)
    
    return train_loader, val_loader, test_loader