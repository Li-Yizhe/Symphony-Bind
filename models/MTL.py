import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from models.heads import ConvBERT


class MTLSharedEncoder(nn.Module):
    """
    Shared encoder for MTL: PLM + ConvBERT (without decoder MLP)
    This extracts features that are shared across all tasks
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # Layer norm for PLM embeddings
        self.layer_norm = nn.LayerNorm(args.hidden_size)
        
        # ConvBERT encoder (without decoder)
        # This is the shared feature extractor
        self.convbert = ConvBERT(
            input_dim=args.hidden_size,
            nhead=getattr(args, 'num_attention_head', 8),
            hidden_dim=getattr(args, 'hidden_dim', 512),
            num_hidden_layers=getattr(args, 'num_hidden_layers', 1),
            kernel_size=getattr(args, 'kernel_size', 7),
            dropout=getattr(args, 'pooling_dropout', 0.2),
            pooling=None,  # No pooling, we need sequence-level features
        )
    
    def plm_embedding(self, plm_model, aa_seq, attention_mask):
        """Extract PLM embeddings"""
        # Check if PLM has trainable parameters (e.g., LoRA)
        # If yes, don't use torch.no_grad() to allow gradient flow
        has_trainable = any(p.requires_grad for p in plm_model.parameters())
        
        if has_trainable:
            # PLM has trainable parameters (LoRA), allow gradients
            if hasattr(plm_model, 'config') and plm_model.config.model_type == 't5':
                outputs = plm_model.encoder(input_ids=aa_seq, attention_mask=attention_mask)
                seq_embeds = outputs.last_hidden_state
            else:
                outputs = plm_model(input_ids=aa_seq, attention_mask=attention_mask)
                seq_embeds = outputs.last_hidden_state
        else:
            # PLM is frozen, use torch.no_grad()
            with torch.no_grad():
                if hasattr(plm_model, 'config') and plm_model.config.model_type == 't5':
                    outputs = plm_model.encoder(input_ids=aa_seq, attention_mask=attention_mask)
                    seq_embeds = outputs.last_hidden_state
                else:
                    outputs = plm_model(input_ids=aa_seq, attention_mask=attention_mask)
                    seq_embeds = outputs.last_hidden_state
        return seq_embeds
    
    def forward(self, plm_model, batch):
        """
        Forward pass through shared encoder
        
        Args:
            plm_model: Pre-trained language model
            batch: Batch dictionary with 'aa_seq_input_ids' and 'aa_seq_attention_mask'
        
        Returns:
            shared_features: [batch_size, seq_len, hidden_size] shared features
        """
        aa_seq = batch['aa_seq_input_ids']
        attention_mask = batch['aa_seq_attention_mask']
        
        # Get PLM embeddings
        seq_embeds = self.plm_embedding(plm_model, aa_seq, attention_mask)
        
        # Apply layer norm
        seq_embeds = self.layer_norm(seq_embeds)
        
        # Apply ConvBERT encoder (without decoder)
        shared_features = self.convbert(seq_embeds, attention_mask)
        
        return shared_features


class MTLTaskDecoder(nn.Module):
    """
    Task-specific decoder for each molecule type
    This is the final MLP layer from ConvBERT decoder
    """
    def __init__(self, hidden_size: int, num_labels: int):
        super().__init__()
        self.num_labels = num_labels
        # Single linear layer decoder (matching ConvBERTClassificationHead decoder)
        self.decoder = nn.Linear(hidden_size, num_labels)
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize decoder parameters"""
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, shared_features, input_mask=None):
        """
        Forward pass through task-specific decoder
        
        Args:
            shared_features: [batch_size, seq_len, hidden_size] shared features
            input_mask: Optional attention mask
        
        Returns:
            logits: [batch_size, seq_len, num_labels] task-specific logits
        """
        batch_size, seq_len, hidden_dim = shared_features.shape
        reshaped_outputs = shared_features.view(-1, hidden_dim)
        logits = self.decoder(reshaped_outputs)
        logits = logits.view(batch_size, seq_len, self.num_labels)
        return logits


class MTLModel(nn.Module):
    """
    Multi-Task Learning Model with Group-Shared Encoders
    - Group-shared encoder: One encoder per group (PLM + ConvBERT without decoder)
    - Task-specific decoders: One MLP decoder per molecule type
    """
    def __init__(self, args, group_to_tasks: dict):
        """
        Initialize MTL model
        
        Args:
            args: Training arguments
            group_to_tasks: Dictionary mapping group names to task names
                e.g., {'nucleotide': ['ADP', 'ATP', 'AMP', 'ANP'], 
                       'inorganic_ion': ['CO', 'PO4', 'SO4'],
                       'cofactor': ['FAD', 'NAD', 'SAH', 'SF4']}
        """
        super().__init__()
        self.args = args
        self.group_to_tasks = group_to_tasks
        self.group_names = list(group_to_tasks.keys())
        self.num_groups = len(self.group_names)
        
        # Flatten all task names
        self.task_names = []
        self.task_name_to_idx = {}
        self.task_name_to_group = {}
        task_idx = 0
        for group_name, tasks in group_to_tasks.items():
            for task_name in tasks:
                self.task_names.append(task_name)
                self.task_name_to_idx[task_name] = task_idx
                self.task_name_to_group[task_name] = group_name
                task_idx += 1
        self.num_tasks = len(self.task_names)
        
        # Create group name to index mapping
        self.group_name_to_idx = {name: idx for idx, name in enumerate(self.group_names)}
        
        # Group-shared encoders (one per group: PLM + ConvBERT without decoder)
        self.group_encoders = nn.ModuleDict({
            group_name: MTLSharedEncoder(args)
            for group_name in self.group_names
        })
        
        # Task-specific decoders (one per molecule type)
        self.task_decoders = nn.ModuleList([
            MTLTaskDecoder(args.hidden_size, args.num_labels)
            for _ in range(self.num_tasks)
        ])
    
    def forward(self, plm_model, batch):
        """
        Forward pass through MTL model
        
        Args:
            plm_model: Pre-trained language model
            batch: Batch dictionary containing:
                - 'aa_seq_input_ids': Tokenized sequences
                - 'aa_seq_attention_mask': Attention masks
                - 'task_id': Task ID for each sample [batch_size]
                - 'group_id': Group ID for each sample [batch_size] (optional, inferred from task_id if not provided)
        
        Returns:
            task_logits: List of logits for each task
                Each element: [batch_size, seq_len, num_labels]
                Note: All tasks return logits for the entire batch, caller should select based on task_id
        """
        batch_size = batch['aa_seq_input_ids'].shape[0]
        task_ids = batch['task_id']  # [batch_size]
        
        # Get group_id for each sample (infer from task_id if not provided)
        if 'group_id' not in batch:
            group_ids = torch.zeros_like(task_ids)
            for i in range(batch_size):
                task_idx = task_ids[i].item()
                task_name = self.task_names[task_idx]
                group_name = self.task_name_to_group[task_name]
                group_ids[i] = self.group_name_to_idx[group_name]
            batch['group_id'] = group_ids
        
        group_ids = batch['group_id']  # [batch_size]
        
        # Process each group separately
        # Group samples by group_id
        group_features = {}
        for group_idx in range(self.num_groups):
            group_mask = (group_ids == group_idx)
            if group_mask.any():
                group_name = self.group_names[group_idx]
                group_encoder = self.group_encoders[group_name]
                
                # Create batch for this group
                group_batch = {
                    'aa_seq_input_ids': batch['aa_seq_input_ids'][group_mask],
                    'aa_seq_attention_mask': batch['aa_seq_attention_mask'][group_mask]
                }
                
                # Get shared features for this group
                group_shared_features = group_encoder(plm_model, group_batch)
                group_features[group_idx] = (group_shared_features, group_mask)
        
        # Generate logits for each task
        # We need to reconstruct the full batch logits
        task_logits = []
        for task_idx in range(self.num_tasks):
            task_name = self.task_names[task_idx]
            group_name = self.task_name_to_group[task_name]
            group_idx = self.group_name_to_idx[group_name]
            
            # Get features for this group
            if group_idx in group_features:
                group_shared_features, group_mask = group_features[group_idx]
                # Apply task-specific decoder to group features
                group_logits = self.task_decoders[task_idx](
                    group_shared_features, 
                    batch['aa_seq_attention_mask'][group_mask] if 'aa_seq_attention_mask' in batch else None
                )
                
                # Create full batch logits (zeros for samples not in this group)
                full_logits = torch.zeros(
                    batch_size, 
                    group_logits.shape[1], 
                    group_logits.shape[2],
                    device=group_logits.device,
                    dtype=group_logits.dtype
                )
                full_logits[group_mask] = group_logits
                task_logits.append(full_logits)
            else:
                # No samples for this group in this batch
                full_logits = torch.zeros(
                    batch_size,
                    batch['aa_seq_input_ids'].shape[1],
                    self.args.num_labels,
                    device=batch['aa_seq_input_ids'].device
                )
                task_logits.append(full_logits)
        
        return task_logits
    
    def get_task_idx(self, task_name: str) -> int:
        """Get task index from task name"""
        return self.task_name_to_idx.get(task_name, 0)

