import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from accelerate import Accelerator

# Add current directory to path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from scheduler import create_scheduler
from metrics import setup_metrics
from losses import MultiClassFocalLossWithAlpha
from models.finetune_models.peft_model_factory import create_plm_and_tokenizer
from peft import PeftModel


class Trainer:
    def __init__(self, args, model, plm_model, logger, train_loader):
        self.args = args
        self.model = model
        self.plm_model = plm_model
        self.logger = logger
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.train_loader = train_loader

        # Setup metrics
        self.metrics_dict = setup_metrics(args)

        # Setup optimizer with different learning rates
        if self.args.training_method in ['plm-lora', 'plm-qlora', 'plm-dora', 'plm-adalora', 'plm-ia3', 'mtl-lora']:
            optimizer_grouped_parameters = [
                {
                    "params": self.model.parameters(),
                    "lr": args.learning_rate
                },
                {
                    "params": [param for param in self.plm_model.parameters() if param.requires_grad],
                    "lr": args.learning_rate
                }
            ]
            self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
        else:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate)

        # Setup accelerator
        self.accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
        
        # Set seed for accelerator to ensure reproducibility
        from accelerate.utils import set_seed as accelerate_set_seed
        accelerate_set_seed(args.seed, device_specific=True)

        # Setup scheduler
        self.scheduler = create_scheduler(args, self.optimizer, self.train_loader)

        # Setup loss function
        self.loss_fn = self._setup_loss_function()

        # Prepare for distributed training
        if self.args.training_method in ['plm-lora', 'plm-qlora', 'plm-dora', 'plm-adalora', 'plm-ia3', 'mtl-lora']:
            self.model, self.plm_model, self.optimizer = self.accelerator.prepare(
                self.model, self.plm_model, self.optimizer
            )
        else:
            self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        if self.scheduler:
            self.scheduler = self.accelerator.prepare(self.scheduler)

        # Training state
        self.best_val_loss = float("inf")
        if self.args.monitor_strategy == 'min':
            self.best_val_metric_score = float("inf")
        else:
            self.best_val_metric_score = -float("inf")
        self.global_steps = 0
        self.early_stop_counter = 0
        self.current_epoch = 0  # Track current epoch for min_epochs check

        # Save args
        with open(os.path.join(self.args.output_dir, f'{self.args.output_model_name.split(".")[0]}.json'), 'w') as f:
            json.dump(self.args.__dict__, f)

    def _setup_loss_function(self):
        if self.args.problem_type == 'single_label_classification' and self.args.num_labels == 2:
            # For binary sequence labeling, use improved loss functions for imbalanced data
            from losses import CombinedLoss, FocalLoss, WeightedBCELoss
            
            # Get loss type from args, default to 'combined'
            loss_type = getattr(self.args, 'loss_type', 'combined')
            
            if loss_type == 'focal':
                # Use only Focal Loss
                return FocalLoss(alpha=1, gamma=2, reduction='mean')
            elif loss_type == 'weighted_bce':
                # Use only Weighted BCE Loss
                return WeightedBCELoss(pos_weight=10.0, reduction='mean')
            elif loss_type == 'combined':
                # Use Combined Loss (default)
                return CombinedLoss(focal_weight=0.7, bce_weight=0.3, alpha=1, gamma=2, pos_weight=10.0)
            else:
                raise ValueError(f"Unsupported loss_type: {loss_type}. Choose from: 'focal', 'weighted_bce', 'combined'")
        else:
            return torch.nn.CrossEntropyLoss()

    def train(self, train_loader, val_loader):
        """Train the model."""
        self.logger.info("=" * 60)
        self.logger.info("Starting Training Phase")
        self.logger.info("=" * 60)
        
        for epoch in range(self.args.num_epochs):
            self.current_epoch = epoch
            self.logger.info(f"---------- Epoch {epoch} ----------")

            # Training phase
            train_loss = self._train_epoch(train_loader)
            self.logger.info(f'Epoch {epoch} Train Loss: {train_loss:.4f}')

            # Validation phase
            val_loss, val_metrics = self._validate(val_loader)

            # Handle validation results (model saving, early stopping)
            self._handle_validation_results(epoch, val_loss, val_metrics)

            # Early stopping check
            if self._check_early_stopping():
                self.logger.info(f"Early stop at Epoch {epoch}")
                break
        
        self.logger.info("=" * 60)
        self.logger.info("Training completed!")
        self.logger.info("=" * 60)

    def _train_epoch(self, train_loader):
        self.model.train()
        if self.args.training_method in ['plm-lora', 'plm-qlora', 'plm-dora', 'plm-adalora', 'plm-ia3', 'mtl-lora']:
            self.plm_model.train()
        total_loss = 0
        total_samples = 0
        epoch_iterator = tqdm(train_loader, desc="Training")

        for batch in epoch_iterator:
            # choose models to accumulate
            models_to_accumulate = [self.model, self.plm_model] if self.args.training_method in ['plm-lora',
                                                                                                 'plm-qlora',
                                                                                                 'plm-dora',
                                                                                                 'plm-adalora',
                                                                                                 'plm-ia3',
                                                                                                 'mtl-lora'] else [
                self.model]

            with self.accelerator.accumulate(*models_to_accumulate):
                # Forward and backward
                loss = self._training_step(batch)
                self.accelerator.backward(loss)

                # Update statistics
                batch_size = batch["label"].size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                # Gradient clipping if needed
                if self.args.max_grad_norm > 0:
                    params_to_clip = (
                        list(self.model.parameters()) + list(self.plm_model.parameters())
                        if self.args.training_method in ['plm-lora', 'plm-qlora', 'plm-dora', 'plm-adalora',
                                                         'plm-ia3', 'mtl-lora']
                        else self.model.parameters()
                    )
                    self.accelerator.clip_grad_norm_(params_to_clip, self.args.max_grad_norm)

                # Optimization step
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()

                # Logging
                self.global_steps += 1
                self._log_training_step(loss)

                # Update progress bar
                epoch_iterator.set_postfix(
                    train_loss=loss.item(),
                    grad_step=self.global_steps // self.args.gradient_accumulation_steps
                )

        return total_loss / total_samples

    def _training_step(self, batch):
        # Move batch to device (only tensors, skip non-tensor values like sequences list)
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Forward pass
        if self.args.training_method in ['mtl', 'mtl-lora']:
            # MTL model returns list of logits for each task
            task_logits = self.model(self.plm_model, batch)
            task_ids = batch['task_id']
            labels = batch["label"]
            
            # Compute loss for each sample based on its task_id
            losses = []
            for i in range(len(task_ids)):
                task_idx = task_ids[i].item()
                sample_logits = task_logits[task_idx][i:i+1]  # [1, seq_len, num_labels]
                sample_labels = labels[i:i+1]  # [1, seq_len]
                sample_loss = self._compute_loss(sample_logits, sample_labels)
                losses.append(sample_loss)
            
            loss = torch.stack(losses).mean()
        else:
            logits = self.model(self.plm_model, batch)
            loss = self._compute_loss(logits, batch["label"])

        return loss

    def _validate(self, val_loader):
        """
        Validate the model.

        Args:
            val_loader: Validation data loader

        Returns:
            tuple: (validation_loss, validation_metrics)
        """
        self.model.eval()
        if self.args.training_method in ['plm-lora', 'plm-qlora', 'plm-dora', 'plm-adalora', 'plm-ia3', 'mtl-lora']:
            self.plm_model.eval()

        total_loss = 0
        total_samples = 0

        # Reset all metrics at the start of validation
        for metric in self.metrics_dict.values():
            if hasattr(metric, 'reset'):
                metric.reset()

        # For MTL, compute loss per task on validation set
        if self.args.training_method in ['mtl', 'mtl-lora']:
            task_names = getattr(self.model, 'task_names', getattr(self.args, 'task_names', []))
            group_to_tasks = getattr(self.model, 'group_to_tasks', {})
            task_name_to_group = getattr(self.model, 'task_name_to_group', {})
            
            # Initialize losses per task and per group
            task_losses = {task_name: [] for task_name in task_names}
            task_samples = {task_name: 0 for task_name in task_names}
            group_losses = {group_name: [] for group_name in group_to_tasks.keys()}
            group_samples = {group_name: 0 for group_name in group_to_tasks.keys()}

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                # Forward pass
                if self.args.training_method in ['mtl', 'mtl-lora']:
                    # MTL model returns list of logits for each task
                    task_logits = self.model(self.plm_model, batch)
                    task_ids = batch['task_id']
                    labels = batch["label"]
                    
                    # Compute loss for each sample based on its task_id
                    for i in range(len(task_ids)):
                        task_idx = task_ids[i].item()
                        task_name = task_names[task_idx]
                        group_name = task_name_to_group.get(task_name, 'unknown')
                        sample_logits = task_logits[task_idx][i:i+1]  # [1, seq_len, num_labels]
                        sample_labels = labels[i:i+1]  # [1, seq_len]
                        sample_loss = self._compute_loss(sample_logits, sample_labels)
                        task_losses[task_name].append(sample_loss.item())
                        task_samples[task_name] += 1
                        group_losses[group_name].append(sample_loss.item())
                        group_samples[group_name] += 1
                    
                    # Compute overall loss for statistics
                    losses = []
                    for i in range(len(task_ids)):
                        task_idx = task_ids[i].item()
                        sample_logits = task_logits[task_idx][i:i+1]
                        sample_labels = labels[i:i+1]
                        sample_loss = self._compute_loss(sample_logits, sample_labels)
                        losses.append(sample_loss)
                    loss = torch.stack(losses).mean()
                    
                    # Don't update global metrics for MTL - we compute per-task metrics separately
                else:
                    logits = self.model(self.plm_model, batch)
                    loss = self._compute_loss(logits, batch["label"])

                # Update loss statistics
                batch_size = len(batch["label"])
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                if self.args.training_method not in ['mtl', 'mtl-lora']:
                    # Update metrics
                    self._update_metrics(logits, batch["label"])

        # Compute average loss
        if self.args.training_method in ['mtl', 'mtl-lora']:
            # For MTL, compute average loss per group on validation set
            group_avg_losses = {group_name: np.mean(losses) if losses else 0.0 
                              for group_name, losses in group_losses.items()}
            # Overall loss is average of group losses
            avg_loss = np.mean(list(group_avg_losses.values()))
            
            # Log per-group validation losses
            self.logger.info("Validation losses per group:")
            for group_name in sorted(group_avg_losses.keys()):
                group_loss = group_avg_losses[group_name]
                group_sample_count = group_samples[group_name]
                self.logger.info(f"  Group {group_name}: {group_loss:.4f} (samples: {group_sample_count})")
                
                # Also log tasks within this group
                tasks_in_group = group_to_tasks.get(group_name, [])
                for task_name in tasks_in_group:
                    if task_name in task_losses and task_losses[task_name]:
                        task_loss = np.mean(task_losses[task_name])
                        task_sample_count = task_samples[task_name]
                        self.logger.info(f"    - {task_name}: {task_loss:.4f} (samples: {task_sample_count})")
        else:
            avg_loss = total_loss / total_samples

        # Compute final metrics
        metrics_results = {}
        
        if self.args.training_method in ['mtl', 'mtl-lora']:
            # For MTL, compute metrics per task on validation set
            task_names = getattr(self.model, 'task_names', getattr(self.args, 'task_names', []))
            
            # Reset metrics for per-task computation
            task_metrics_dict = {}
            for task_name in task_names:
                task_metrics_dict[task_name] = {}
                for metric_name in self.metrics_dict.keys():
                    # Create a new metric instance for each task
                    from metrics import setup_metrics
                    import types
                    temp_args = types.SimpleNamespace()
                    temp_args.problem_type = self.args.problem_type
                    temp_args.num_labels = self.args.num_labels
                    temp_args.metrics = [metric_name]
                    task_metric_dict = setup_metrics(temp_args)
                    if metric_name in task_metric_dict:
                        task_metrics_dict[task_name][metric_name] = task_metric_dict[metric_name]
            
            # Re-compute metrics per task
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    
                    task_logits = self.model(self.plm_model, batch)
                    task_ids = batch['task_id']
                    labels = batch["label"]
                    
                    # Update metrics per task
                    for i in range(len(task_ids)):
                        task_idx = task_ids[i].item()
                        task_name = task_names[task_idx]
                        sample_logits = task_logits[task_idx][i:i+1]
                        sample_labels = labels[i:i+1]
                        
                        # Update task-specific metrics
                        for metric_name, metric in task_metrics_dict[task_name].items():
                            batch_size, seq_len, num_classes = sample_logits.shape
                            logits_flat = sample_logits.view(-1, num_classes)
                            labels_flat = sample_labels.view(-1)
                            
                            if metric_name in ['auroc', 'aupr']:
                                probs = torch.softmax(logits_flat, dim=1)[:, 1]
                                metric(probs, labels_flat)
                            elif metric_name == 'mcc':
                                preds = torch.argmax(logits_flat, dim=1)
                                metric(preds, labels_flat)
            
            # Compute final metrics per task and organize by group
            group_to_tasks = getattr(self.model, 'group_to_tasks', {})
            task_name_to_group = getattr(self.model, 'task_name_to_group', {})
            
            # Store metrics per task
            for task_name in task_names:
                for metric_name, metric in task_metrics_dict[task_name].items():
                    if hasattr(metric, 'compute'):
                        metric_value = metric.compute().item()
                        metrics_results[f"{task_name}_{metric_name}"] = metric_value
            
            # Log metrics per group
            self.logger.info("Per-group validation metrics:")
            for group_name in sorted(group_to_tasks.keys()):
                tasks_in_group = group_to_tasks.get(group_name, [])
                self.logger.info(f"  Group {group_name}:")
                for task_name in tasks_in_group:
                    task_metrics = {}
                    for metric_name in self.metrics_dict.keys():
                        metric_key = f"{task_name}_{metric_name}"
                        if metric_key in metrics_results:
                            task_metrics[metric_name] = metrics_results[metric_key]
                    if task_metrics:
                        metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in task_metrics.items()])
                        self.logger.info(f"    - {task_name}: {metric_str}")
        else:
            for name, metric in self.metrics_dict.items():
                if hasattr(metric, 'compute'):
                    metrics_results[name] = metric.compute().item()

        return avg_loss, metrics_results

    def _test_evaluate(self, test_loader):
        """Test evaluation - same as _validate but with 'Testing' progress bar."""
        self.model.eval()
        if self.args.training_method in ['plm-lora', 'plm-qlora', 'plm-dora', 'plm-adalora', 'plm-ia3', 'mtl-lora']:
            self.plm_model.eval()

        total_loss = 0
        total_samples = 0

        # Reset all metrics at the start of testing
        for metric in self.metrics_dict.values():
            if hasattr(metric, 'reset'):
                metric.reset()

        # For MTL, compute loss per task and per group on test set
        if self.args.training_method in ['mtl', 'mtl-lora']:
            task_names = getattr(self.model, 'task_names', getattr(self.args, 'task_names', []))
            group_to_tasks = getattr(self.model, 'group_to_tasks', {})
            task_name_to_group = getattr(self.model, 'task_name_to_group', {})
            
            # Initialize losses per task and per group
            task_losses = {task_name: [] for task_name in task_names}
            task_samples = {task_name: 0 for task_name in task_names}
            group_losses = {group_name: [] for group_name in group_to_tasks.keys()}
            group_samples = {group_name: 0 for group_name in group_to_tasks.keys()}

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                # Forward pass
                if self.args.training_method in ['mtl', 'mtl-lora']:
                    # MTL model returns list of logits for each task
                    task_logits = self.model(self.plm_model, batch)
                    task_ids = batch['task_id']
                    labels = batch["label"]
                    
                    # Compute loss for each sample based on its task_id
                    for i in range(len(task_ids)):
                        task_idx = task_ids[i].item()
                        task_name = task_names[task_idx]
                        group_name = task_name_to_group.get(task_name, 'unknown')
                        sample_logits = task_logits[task_idx][i:i+1]
                        sample_labels = labels[i:i+1]
                        sample_loss = self._compute_loss(sample_logits, sample_labels)
                        task_losses[task_name].append(sample_loss.item())
                        task_samples[task_name] += 1
                        group_losses[group_name].append(sample_loss.item())
                        group_samples[group_name] += 1
                    
                    # Compute overall loss for statistics
                    losses = []
                    for i in range(len(task_ids)):
                        task_idx = task_ids[i].item()
                        sample_logits = task_logits[task_idx][i:i+1]
                        sample_labels = labels[i:i+1]
                        sample_loss = self._compute_loss(sample_logits, sample_labels)
                        losses.append(sample_loss)
                    loss = torch.stack(losses).mean()
                    
                    # Don't update global metrics for MTL - we compute per-task metrics separately
                else:
                    logits = self.model(self.plm_model, batch)
                    loss = self._compute_loss(logits, batch["label"])

                # Update loss statistics
                batch_size = len(batch["label"])
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                if self.args.training_method not in ['mtl', 'mtl-lora']:
                    # Update metrics
                    self._update_metrics(logits, batch["label"])

        # Compute average loss
        if self.args.training_method in ['mtl', 'mtl-lora']:
            # For MTL, compute average loss per group on test set
            group_avg_losses = {group_name: np.mean(losses) if losses else 0.0 
                              for group_name, losses in group_losses.items()}
            # Overall loss is average of group losses
            avg_loss = np.mean(list(group_avg_losses.values()))
            
            # Log per-group test losses
            self.logger.info("Test losses per group:")
            for group_name in sorted(group_avg_losses.keys()):
                group_loss = group_avg_losses[group_name]
                group_sample_count = group_samples[group_name]
                self.logger.info(f"  Group {group_name}: {group_loss:.4f} (samples: {group_sample_count})")
                
                # Also log tasks within this group
                tasks_in_group = group_to_tasks.get(group_name, [])
                for task_name in tasks_in_group:
                    if task_name in task_losses and task_losses[task_name]:
                        task_loss = np.mean(task_losses[task_name])
                        task_sample_count = task_samples[task_name]
                        self.logger.info(f"    - {task_name}: {task_loss:.4f} (samples: {task_sample_count})")
        else:
            avg_loss = total_loss / total_samples

        # Compute final metrics
        metrics_results = {}
        
        if self.args.training_method in ['mtl', 'mtl-lora']:
            # For MTL, compute metrics per task on test set
            task_names = getattr(self.model, 'task_names', getattr(self.args, 'task_names', []))
            
            # Reset metrics for per-task computation
            task_metrics_dict = {}
            for task_name in task_names:
                task_metrics_dict[task_name] = {}
                for metric_name in self.metrics_dict.keys():
                    # Create a new metric instance for each task
                    from metrics import setup_metrics
                    import types
                    temp_args = types.SimpleNamespace()
                    temp_args.problem_type = self.args.problem_type
                    temp_args.num_labels = self.args.num_labels
                    temp_args.metrics = [metric_name]
                    task_metric_dict = setup_metrics(temp_args)
                    if metric_name in task_metric_dict:
                        task_metrics_dict[task_name][metric_name] = task_metric_dict[metric_name]
            
            # Re-compute metrics per task
            with torch.no_grad():
                for batch in test_loader:
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    
                    task_logits = self.model(self.plm_model, batch)
                    task_ids = batch['task_id']
                    labels = batch["label"]
                    
                    # Update metrics per task
                    for i in range(len(task_ids)):
                        task_idx = task_ids[i].item()
                        task_name = task_names[task_idx]
                        sample_logits = task_logits[task_idx][i:i+1]
                        sample_labels = labels[i:i+1]
                        
                        # Update task-specific metrics
                        for metric_name, metric in task_metrics_dict[task_name].items():
                            batch_size, seq_len, num_classes = sample_logits.shape
                            logits_flat = sample_logits.view(-1, num_classes)
                            labels_flat = sample_labels.view(-1)
                            
                            if metric_name in ['auroc', 'aupr']:
                                probs = torch.softmax(logits_flat, dim=1)[:, 1]
                                metric(probs, labels_flat)
                            elif metric_name == 'mcc':
                                preds = torch.argmax(logits_flat, dim=1)
                                metric(preds, labels_flat)
            
            # Compute final metrics per task and organize by group
            group_to_tasks = getattr(self.model, 'group_to_tasks', {})
            task_name_to_group = getattr(self.model, 'task_name_to_group', {})
            
            # Store metrics per task
            for task_name in task_names:
                for metric_name, metric in task_metrics_dict[task_name].items():
                    if hasattr(metric, 'compute'):
                        metric_value = metric.compute().item()
                        metrics_results[f"{task_name}_{metric_name}"] = metric_value
            
            # Log metrics per group
            self.logger.info("Per-group test metrics:")
            for group_name in sorted(group_to_tasks.keys()):
                tasks_in_group = group_to_tasks.get(group_name, [])
                self.logger.info(f"  Group {group_name}:")
                for task_name in tasks_in_group:
                    task_metrics = {}
                    for metric_name in self.metrics_dict.keys():
                        metric_key = f"{task_name}_{metric_name}"
                        if metric_key in metrics_results:
                            task_metrics[metric_name] = metrics_results[metric_key]
                    if task_metrics:
                        metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in task_metrics.items()])
                        self.logger.info(f"    - {task_name}: {metric_str}")
        else:
            for name, metric in self.metrics_dict.items():
                if hasattr(metric, 'compute'):
                    metrics_results[name] = metric.compute().item()

        return avg_loss, metrics_results

    def test(self, test_loader, tokenizer=None):
        # Load best model
        self.logger.info("=" * 60)
        self.logger.info("Starting Test Phase")
        self.logger.info("=" * 60)
        self._load_best_model()

        # Run evaluation
        test_loss, test_metrics = self._test_evaluate(test_loader)

        # Log results
        self.logger.info("=" * 60)
        self.logger.info("Test Results:")
        self.logger.info(f"Test Loss: {test_loss:.4f}")
        
        if self.args.training_method in ['mtl', 'mtl-lora']:
            # Log per-group metrics (already logged in _test_evaluate, but also log here for consistency)
            group_to_tasks = getattr(self.model, 'group_to_tasks', {})
            task_names = getattr(self.model, 'task_names', getattr(self.args, 'task_names', []))
            self.logger.info("Per-group test metrics:")
            for group_name in sorted(group_to_tasks.keys()):
                tasks_in_group = group_to_tasks.get(group_name, [])
                self.logger.info(f"  Group {group_name}:")
                for task_name in tasks_in_group:
                    task_metrics = {}
                    for metric_name in ['mcc', 'auroc', 'aupr']:
                        key = f"{task_name}_{metric_name}"
                        if key in test_metrics:
                            task_metrics[metric_name] = test_metrics[key]
                    if task_metrics:
                        metric_str = ", ".join([f"{k.upper()}: {v:.4f}" for k, v in task_metrics.items()])
                        self.logger.info(f"    - {task_name}: {metric_str}")
        else:
            for name, value in test_metrics.items():
                self.logger.info(f"Test {name}: {value:.4f}")
        
        self.logger.info("=" * 60)

        # Save test results to JSON file
        test_results = {
            "test_loss": float(test_loss),
            **{f"test_{k}": float(v) for k, v in test_metrics.items()}
        }
        results_file = os.path.join(self.args.output_dir, "test_results.json")
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=4)
        self.logger.info(f"Test results saved to: {results_file}")

        # Save detailed predictions
        self._save_detailed_predictions(test_loader, tokenizer)

        if self.args.wandb:
            import wandb
            wandb.log({f"test/{k}": v for k, v in test_metrics.items()})
            wandb.log({"test/loss": test_loss})

    def _compute_loss(self, logits, labels):
        if self.args.problem_type == 'single_label_classification' and self.args.num_labels == 2:
            # For binary sequence labeling, reshape logits and labels
            # logits: [batch_size, seq_len, 2] -> [batch_size * seq_len, 2]
            # labels: [batch_size, seq_len] -> [batch_size * seq_len]
            batch_size, seq_len, num_classes = logits.shape
            logits = logits.view(-1, num_classes)  # [batch_size * seq_len, 2]
            labels = labels.view(-1).long()  # [batch_size * seq_len]
            return self.loss_fn(logits, labels)
        else:
            return self.loss_fn(logits, labels)

    def _update_metrics(self, logits, labels):
        """Update metrics with current batch predictions."""
        # For sequence labeling, we need to flatten the predictions and labels
        # logits: [batch_size, seq_len, 2] -> [batch_size * seq_len, 2]
        # labels: [batch_size, seq_len] -> [batch_size * seq_len]
        batch_size, seq_len, num_classes = logits.shape
        logits_flat = logits.view(-1, num_classes)  # [batch_size * seq_len, 2]
        labels_flat = labels.view(-1)  # [batch_size * seq_len]
        
        for metric_name, metric in self.metrics_dict.items():
            if metric_name in ['auroc', 'aupr']:
                # For AUROC and AUPR, we need the probability of positive class
                probs = torch.softmax(logits_flat, dim=1)[:, 1]  # Probability of class 1
                metric(probs, labels_flat)
            elif metric_name == 'mcc':
                # For MCC, we need the predicted class
                preds = torch.argmax(logits_flat, dim=1)
                metric(preds, labels_flat)

    def _log_training_step(self, loss):
        if self.args.wandb:
            import wandb
            wandb.log({
                "train/loss": loss.item(),
                "train/learning_rate": self.optimizer.param_groups[0]['lr']
            }, step=self.global_steps)

    def _save_model(self, path):
        if self.args.training_method in ["plm-lora", "mtl-lora"]:
            model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
            torch.save(model_state, path)
            plm_lora_path = path.replace('.pt', '_lora')
            self.plm_model.save_pretrained(plm_lora_path)
        elif self.args.training_method == "plm-qlora":
            # save model state dict
            model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
            torch.save(model_state, path)
            plm_qlora_path = path.replace('.pt', '_qlora')
            # save plm model lora weights
            self.plm_model.save_pretrained(plm_qlora_path)
        elif self.args.training_method == "plm-dora":
            # save model state dict
            model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
            torch.save(model_state, path)
            plm_dora_path = path.replace('.pt', '_dora')
            # save plm model lora weights
            self.plm_model.save_pretrained(plm_dora_path)
        elif self.args.training_method == "plm-adalora":
            # save model state dict
            model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
            torch.save(model_state, path)
            plm_adalora_path = path.replace('.pt', '_adalora')
            self.plm_model.save_pretrained(plm_adalora_path)
        elif self.args.training_method == "plm-ia3":
            model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
            torch.save(model_state, path)
            plm_ia3_path = path.replace('.pt', '_ia3')
            self.plm_model.save_pretrained(plm_ia3_path)
        else:
            model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
            torch.save(model_state, path)

    def _clean_peft_config(self, peft_path):
        """Clean PEFT config file by removing incompatible parameters."""
        adapter_config_path = os.path.join(peft_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            import json
            with open(adapter_config_path, 'r') as f:
                config = json.load(f)
            
            # Remove incompatible parameters
            incompatible_params = ['corda_config', 'cordalora_config', 'cordalora_alpha', 'eva_config']
            params_removed = []
            for param in incompatible_params:
                if param in config:
                    params_removed.append(param)
                    del config[param]
            
            if params_removed:
                self.logger.warning(f"Removed incompatible PEFT parameters: {', '.join(params_removed)}")
                # ✅ DO NOT save - just log the warning
                # These parameters are only incompatible at load time, removing them permanently 
                # would destroy the original training configuration
                # with open(adapter_config_path, 'w') as f:
                #     json.dump(config, f, indent=2)

    def _load_best_model(self):
        path = os.path.join(self.args.output_dir, self.args.output_model_name)
        if self.args.training_method in ["plm-lora", "mtl-lora"]:
            checkpoint = torch.load(path, map_location="cpu")
            self.model.load_state_dict(checkpoint)
            plm_lora_path = path.replace('.pt', '_lora')
            _, self.plm_model = create_plm_and_tokenizer(self.args)
            self._clean_peft_config(plm_lora_path)
            self.plm_model = PeftModel.from_pretrained(self.plm_model, plm_lora_path)
            self.plm_model = self.plm_model.merge_and_unload()
            self.model.to(self.device)
            self.plm_model.to(self.device)
            self.model.eval()
            self.plm_model.eval()
        elif self.args.training_method == "plm-qlora":
            # load model state dict
            checkpoint = torch.load(path, map_location="cpu")
            self.model.load_state_dict(checkpoint)
            plm_qlora_path = path.replace('.pt', '_qlora')
            # reload plm model and apply qlora weights
            _, self.plm_model = create_plm_and_tokenizer(self.args)
            self._clean_peft_config(plm_qlora_path)
            self.plm_model = PeftModel.from_pretrained(self.plm_model, plm_qlora_path)
            self.plm_model = self.plm_model.merge_and_unload()
            self.model.to(self.device)
            self.plm_model.to(self.device)
            self.model.eval()
            self.plm_model.eval()
        elif self.args.training_method == "plm-dora":
            # load model state dict
            checkpoint = torch.load(path, map_location="cpu")
            self.model.load_state_dict(checkpoint)
            plm_dora_path = path.replace('.pt', '_dora')
            # reload plm model and apply dora weights
            _, self.plm_model = create_plm_and_tokenizer(self.args)
            self._clean_peft_config(plm_dora_path)
            self.plm_model = PeftModel.from_pretrained(self.plm_model, plm_dora_path)
            self.plm_model = self.plm_model.merge_and_unload()
            self.model.to(self.device)
            self.plm_model.to(self.device)
            self.model.eval()
            self.plm_model.eval()
        elif self.args.training_method == "plm-adalora":
            # load model state dict
            checkpoint = torch.load(path, map_location="cpu")
            self.model.load_state_dict(checkpoint)
            plm_adalora_path = path.replace('.pt', '_adalora')
            # reload plm model and apply adalora weights
            _, self.plm_model = create_plm_and_tokenizer(self.args)
            self._clean_peft_config(plm_adalora_path)
            self.plm_model = PeftModel.from_pretrained(self.plm_model, plm_adalora_path)
            self.plm_model = self.plm_model.merge_and_unload()
            self.model.to(self.device)
            self.plm_model.to(self.device)
            self.model.eval()
            self.plm_model.eval()
        elif self.args.training_method == "plm-ia3":
            checkpoint = torch.load(path, map_location="cpu")
            self.model.load_state_dict(checkpoint)
            plm_ia3_path = path.replace('.pt', '_ia3')
            _, self.plm_model = create_plm_and_tokenizer(self.args)
            self._clean_peft_config(plm_ia3_path)
            self.plm_model = PeftModel.from_pretrained(self.plm_model, plm_ia3_path)
            self.plm_model = self.plm_model.merge_and_unload()
            self.model.to(self.device)
            self.plm_model.to(self.device)
            self.model.eval()
            self.plm_model.eval()
        else:
            checkpoint = torch.load(path, map_location="cpu")
            self.model.load_state_dict(checkpoint)
            self.model.to(self.device)
            self.model.eval()

    def _handle_validation_results(self, epoch: int, val_loss: float, val_metrics: dict):
        """
        Handle validation results, including model saving and early stopping checks.

        Args:
            epoch: Current epoch number
            val_loss: Validation loss
            val_metrics: Dictionary of validation metrics
        """
        # Log validation results
        self.logger.info(f'Epoch {epoch} Val Loss: {val_loss:.4f}')
        for metric_name, metric_value in val_metrics.items():
            self.logger.info(f'Epoch {epoch} Val {metric_name}: {metric_value:.4f}')

        if self.args.wandb:
            import wandb
            wandb.log({
                "val/loss": val_loss,
                **{f"val/{k}": v for k, v in val_metrics.items()}
            }, step=self.global_steps)

        # Track best val_loss for reference (no early stopping based on val_loss)
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss

        # Check if we should save the model
        should_save = False
        monitor_value = val_loss

        # If monitoring a specific metric
        if self.args.monitor != 'loss':
            if self.args.training_method in ['mtl', 'mtl-lora']:
                # For MTL, compute average metric across all tasks in the group
                task_names = getattr(self.model, 'task_names', getattr(self.args, 'task_names', []))
                monitor_values = []
                for task_name in task_names:
                    metric_key = f"{task_name}_{self.args.monitor}"
                    if metric_key in val_metrics:
                        monitor_values.append(val_metrics[metric_key])
                if monitor_values:
                    monitor_value = np.mean(monitor_values)
                else:
                    # Fallback to loss if metric not found
                    monitor_value = val_loss
            elif self.args.monitor in val_metrics:
                monitor_value = val_metrics[self.args.monitor]

        # Check if current result is better with min_delta consideration
        if self.args.monitor_strategy == 'min':
            if monitor_value < self.best_val_metric_score - self.args.min_delta:
                should_save = True
                self.best_val_metric_score = monitor_value
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
        else:  # strategy == 'max'
            if monitor_value > self.best_val_metric_score + self.args.min_delta:
                should_save = True
                self.best_val_metric_score = monitor_value
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1

        # Save model if improved
        if should_save:
            self.logger.info(f"Saving model with best val {self.args.monitor}: {monitor_value:.4f}")
            save_path = os.path.join(self.args.output_dir, self.args.output_model_name)
            self._save_model(save_path)

    def _check_early_stopping(self) -> bool:
        """
        Check if training should be stopped early.

        Returns:
            bool: True if training should stop, False otherwise
        """
        # Check if minimum epochs have been reached
        if self.current_epoch < self.args.min_epochs:
            return False
            
        # Check if monitored metric (AUPR) hasn't improved for patience epochs
        if self.args.patience > 0 and self.early_stop_counter >= self.args.patience:
            self.logger.info(f"Early stopping triggered after {self.early_stop_counter} epochs without improvement in {self.args.monitor} (threshold: {self.args.min_delta})")
            return True
            
        return False

    def _save_detailed_predictions(self, test_loader, tokenizer=None):
        """
        Save detailed predictions with TP/FP/FN/TN classification for each position.
        """
        self.model.eval()
        if self.args.training_method in ['plm-lora', 'plm-qlora', 'plm-dora', 'plm-adalora', 'plm-ia3', 'mtl-lora']:
            self.plm_model.eval()

        predictions_data = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Collecting predictions"):
                # Separate sequences (strings) from tensors
                sequences = batch["sequences"]
                batch_device = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                               for k, v in batch.items()}
                
                # Forward pass
                if self.args.training_method in ['mtl', 'mtl-lora']:
                    # MTL model returns list of logits for each task
                    task_logits = self.model(self.plm_model, batch_device)
                    task_ids = batch_device['task_id']
                    labels = batch_device["label"]
                    attention_mask = batch_device.get("aa_seq_attention_mask", None)
                    
                    # Get predictions for each sample based on its task_id
                    preds_list = []
                    for i in range(len(task_ids)):
                        task_idx = task_ids[i].item()
                        sample_logits = task_logits[task_idx][i:i+1]  # [1, seq_len, num_labels]
                        sample_preds = torch.argmax(sample_logits, dim=-1)  # [1, seq_len]
                        preds_list.append(sample_preds.squeeze(0))
                    preds = torch.stack(preds_list)  # [batch_size, seq_len]
                else:
                    logits = self.model(self.plm_model, batch_device)
                    preds = torch.argmax(logits, dim=-1)  # [batch_size, seq_len]
                    labels = batch_device["label"]  # [batch_size, seq_len]
                    attention_mask = batch_device.get("aa_seq_attention_mask", None)
                
                # Move to CPU for processing
                preds_cpu = preds.cpu().numpy()
                labels_cpu = labels.cpu().numpy()
                if attention_mask is not None:
                    attention_mask_cpu = attention_mask.cpu().numpy()
                else:
                    attention_mask_cpu = None
                
                # Get task information for MTL
                task_names_list = None
                task_ids_cpu = None
                if self.args.training_method in ['mtl', 'mtl-lora']:
                    task_names_list = getattr(self.model, 'task_names', getattr(self.args, 'task_names', []))
                    task_ids_cpu = batch_device['task_id'].cpu().numpy()
                
                # Process each sequence in batch
                for i in range(len(labels_cpu)):
                    seq_preds = preds_cpu[i]
                    seq_labels = labels_cpu[i]
                    original_sequence = sequences[i]
                    
                    # Get task name for MTL
                    task_name = None
                    if self.args.training_method in ['mtl', 'mtl-lora'] and task_names_list:
                        task_idx = task_ids_cpu[i]
                        task_name = task_names_list[task_idx]
                    
                    # Filter out special tokens using attention mask
                    # The aligned labels should match the tokenized length
                    # We need to extract only the predictions for actual amino acids
                    if attention_mask_cpu is not None:
                        mask = attention_mask_cpu[i]
                        # Get valid positions (non-padded)
                        valid_positions = mask == 1
                        seq_preds_valid = seq_preds[valid_positions]
                        seq_labels_valid = seq_labels[valid_positions]
                    else:
                        seq_preds_valid = seq_preds
                        seq_labels_valid = seq_labels
                    
                    # Match predictions to original sequence length
                    # Remove special tokens (CLS, SEP/EOS) - typically first and last
                    seq_len = len(original_sequence)
                    if len(seq_preds_valid) > seq_len:
                        # Remove special tokens (assume first and last)
                        if len(seq_preds_valid) == seq_len + 2:
                            seq_preds_valid = seq_preds_valid[1:-1]
                            seq_labels_valid = seq_labels_valid[1:-1]
                        elif len(seq_preds_valid) == seq_len + 1:
                            # Only one special token
                            seq_preds_valid = seq_preds_valid[:-1]
                            seq_labels_valid = seq_labels_valid[:-1]
                    
                    # Truncate to sequence length if still longer
                    seq_preds_valid = seq_preds_valid[:seq_len]
                    seq_labels_valid = seq_labels_valid[:seq_len]
                    
                    # Pad if shorter
                    if len(seq_preds_valid) < seq_len:
                        padding_len = seq_len - len(seq_preds_valid)
                        seq_preds_valid = list(seq_preds_valid) + [0] * padding_len
                        seq_labels_valid = list(seq_labels_valid) + [0] * padding_len
                    
                    # Convert to lists if numpy arrays
                    if isinstance(seq_preds_valid, np.ndarray):
                        seq_preds_valid = seq_preds_valid.tolist()
                    if isinstance(seq_labels_valid, np.ndarray):
                        seq_labels_valid = seq_labels_valid.tolist()
                    
                    # Convert predictions and labels to strings
                    pred_str = ''.join(map(str, seq_preds_valid))
                    label_str = ''.join(map(str, seq_labels_valid))
                    
                    # Calculate TP, FP, FN, TN counts for this sequence
                    tp_count = 0
                    fp_count = 0
                    fn_count = 0
                    tn_count = 0
                    
                    for pred, label in zip(seq_preds_valid, seq_labels_valid):
                        if pred == 1 and label == 1:
                            tp_count += 1
                        elif pred == 1 and label == 0:
                            fp_count += 1
                        elif pred == 0 and label == 1:
                            fn_count += 1
                        else:  # pred == 0 and label == 0
                            tn_count += 1
                    
                    pred_dict = {
                        'sequence': original_sequence,
                        'true_label': label_str,
                        'prediction': pred_str,
                        'TP': tp_count,
                        'FP': fp_count,
                        'FN': fn_count,
                        'TN': tn_count
                    }
                    
                    # Add task name for MTL
                    if task_name is not None:
                        pred_dict['task_name'] = task_name
                    
                    predictions_data.append(pred_dict)
        
        # Extract dataset name from test_file path or use MTL task names
        if self.args.training_method in ['mtl', 'mtl-lora']:
            task_names = getattr(self.model, 'task_names', getattr(self.args, 'task_names', []))
            if task_names:
                dataset_name = '_'.join(task_names)
            else:
                dataset_name = "mtl_all_tasks"
        elif hasattr(self.args, 'test_file') and self.args.test_file:
            dataset_name = os.path.basename(os.path.dirname(self.args.test_file))
        else:
            dataset_name = "unknown"
        
        # Save to CSV file with @test format
        output_file = os.path.join(self.args.output_dir, f"{dataset_name}@test.csv")
        
        df = pd.DataFrame(predictions_data)
        df.to_csv(output_file, index=False)
        
        self.logger.info(f"Detailed predictions saved to: {output_file}")
        self.logger.info(f"Total sequences predicted: {len(predictions_data)}")