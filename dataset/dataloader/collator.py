import torch
from typing import Dict, List, Any
from transformers import PreTrainedTokenizer
from dataclasses import dataclass

@dataclass
class Collator:
    """Data collator class for protein sequence classification with proper label alignment."""
    tokenizer: PreTrainedTokenizer
    max_length: int = None
    problem_type: str = 'classification'
    num_labels: int = None

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate function for batching examples with proper label alignment."""
        # Initialize lists to store sequences and labels
        aa_seqs, labels = [], []

        # Process each example
        for e in examples:
            # Process sequences
            aa_seqs.append(e["sequence"])
            
            # Process labels - convert binary string to list of integers
            if isinstance(e["label"], str):
                # Convert binary string like "0001001..." to list of integers
                label_sequence = [int(bit) for bit in e["label"]]
                e["label"] = label_sequence
            
            labels.append(e["label"])

        # Tokenize sequences
        batch = self.tokenize_sequences(aa_seqs)
        
        # CRITICAL FIX: Properly align labels with tokens
        aligned_labels = self.align_labels_with_tokens(aa_seqs, labels, batch["aa_seq_input_ids"])
        
        batch["label"] = torch.as_tensor(
            aligned_labels, 
            dtype=torch.float if self.problem_type == 'regression' else torch.long
        )
        
        # Store original sequences for prediction output
        batch["sequences"] = aa_seqs

        return batch
    
    def align_labels_with_tokens(self, sequences, labels, input_ids):
        """Properly align labels with tokenized sequences."""
        aligned_labels = []
        
        for seq, label, token_ids in zip(sequences, labels, input_ids):
            # For protein models, each amino acid should correspond to one label
            # Check if this is a protein tokenizer (T5, ProtBert, etc.)
            is_protein_tokenizer = (
                hasattr(self.tokenizer, '__class__') and 
                ('T5' in self.tokenizer.__class__.__name__ or 
                 'ProtBert' in str(self.tokenizer) or
                 'ESM' in str(self.tokenizer))
            )
            
            if is_protein_tokenizer:
                # For protein tokenizers, align 1:1 with sequence characters
                aligned_label = []
                seq_idx = 0
                
                for i, token_id in enumerate(token_ids):
                    # Skip special tokens (CLS, SEP, PAD)
                    if (token_id == self.tokenizer.cls_token_id or 
                        token_id == self.tokenizer.sep_token_id or 
                        token_id == self.tokenizer.pad_token_id or
                        token_id == self.tokenizer.eos_token_id):
                        aligned_label.append(0)  # Special token
                    else:
                        # Regular amino acid token
                        if seq_idx < len(label):
                            aligned_label.append(label[seq_idx])
                            seq_idx += 1
                        else:
                            aligned_label.append(0)  # Padding
                
                # Ensure length matches
                aligned_label = aligned_label[:len(token_ids)]
            else:
                # For other tokenizers, use the original logic
                tokens = self.tokenizer.tokenize(seq)
                aligned_label = [0]  # CLS token
                
                seq_idx = 0
                for token in tokens:
                    if token.startswith('##') or token.startswith('▁'):
                        # Subword token, use previous label
                        if seq_idx < len(label):
                            aligned_label.append(label[seq_idx])
                    else:
                        # New token, advance sequence index
                        if seq_idx < len(label):
                            aligned_label.append(label[seq_idx])
                            seq_idx += 1
                        else:
                            aligned_label.append(0)  # Padding
                
                # Add SEP token if present
                if len(aligned_label) < len(token_ids):
                    aligned_label.append(0)  # SEP token
                
                # Pad to match input_ids length
                while len(aligned_label) < len(token_ids):
                    aligned_label.append(0)  # PAD tokens
                    
                # Truncate if too long
                aligned_label = aligned_label[:len(token_ids)]
            
            aligned_labels.append(aligned_label)
        
        return aligned_labels
    
    def tokenize_sequences(self, aa_seqs: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize amino acid sequences."""
        # Check if this is a protein tokenizer that needs special handling
        is_protein_tokenizer = (
            hasattr(self.tokenizer, '__class__') and 
            ('T5' in self.tokenizer.__class__.__name__ or 
             'ProtBert' in str(self.tokenizer) or
             'ESM' in str(self.tokenizer))
        )
        
        if is_protein_tokenizer:
            # For protein tokenizers, manually tokenize each amino acid
            input_ids_list = []
            attention_masks_list = []
            
            for seq in aa_seqs:
                # Convert single letter amino acids to tokens
                tokens = []
                for char in seq:
                    if char in self._get_aa_mapping():
                        three_letter = self._get_aa_mapping()[char]
                        aa_token = f'▁{three_letter}'
                        tokens.append(aa_token)
                    else:
                        tokens.append('▁<unk>')
                
                # Convert tokens to IDs
                token_ids = []
                for token in tokens:
                    token_ids.append(self.tokenizer.get_vocab().get(token, self.tokenizer.unk_token_id))
                
                # Add special tokens if needed
                if hasattr(self.tokenizer, 'bos_token_id') and self.tokenizer.bos_token_id is not None:
                    token_ids = [self.tokenizer.bos_token_id] + token_ids
                if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
                    token_ids = token_ids + [self.tokenizer.eos_token_id]
                
                # Truncate if too long
                if self.max_length and len(token_ids) > self.max_length:
                    token_ids = token_ids[:self.max_length]
                
                input_ids_list.append(token_ids)
                attention_masks_list.append([1] * len(token_ids))
            
            # Pad sequences
            max_len = max(len(ids) for ids in input_ids_list)
            if self.max_length:
                max_len = min(max_len, self.max_length)
            
            padded_input_ids = []
            padded_attention_masks = []
            
            for ids, mask in zip(input_ids_list, attention_masks_list):
                # Pad with pad_token_id
                pad_length = max_len - len(ids)
                padded_ids = ids + [self.tokenizer.pad_token_id] * pad_length
                padded_mask = mask + [0] * pad_length
                
                padded_input_ids.append(padded_ids)
                padded_attention_masks.append(padded_mask)
            
            batch = {
                "aa_seq_input_ids": torch.tensor(padded_input_ids),
                "aa_seq_attention_mask": torch.tensor(padded_attention_masks)
            }
        else:
            # For other tokenizers, use standard tokenization
            aa_encodings = self.tokenizer(
                aa_seqs,
                padding=True,
                truncation=True if self.max_length else False,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            batch = {
                "aa_seq_input_ids": aa_encodings["input_ids"],
                "aa_seq_attention_mask": aa_encodings["attention_mask"]
            }
        
        return batch
    
    def _get_aa_mapping(self):
        """Get amino acid single letter to three letter mapping."""
        return {
            'A': 'Ala', 'R': 'Arg', 'N': 'Asn', 'D': 'Asp', 'C': 'Cys',
            'Q': 'Gln', 'E': 'Glu', 'G': 'Gly', 'H': 'His', 'I': 'Ile',
            'L': 'Leu', 'K': 'Lys', 'M': 'Met', 'F': 'Phe', 'P': 'Pro',
            'S': 'Ser', 'T': 'Thr', 'W': 'Trp', 'Y': 'Tyr', 'V': 'Val'
        }


@dataclass
class MTLCollator(Collator):
    """Data collator for MTL that includes task_id and group_id in batch."""
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate function for MTL batching with task_id and group_id."""
        # Get task_ids and group_ids from examples
        task_ids = [e.get('task_id', 0) for e in examples]
        group_ids = [e.get('group_id', 0) for e in examples]
        
        # Call parent collator to get standard batch
        batch = super().__call__(examples)
        
        # Add task_id and group_id to batch
        batch['task_id'] = torch.tensor(task_ids, dtype=torch.long)
        batch['group_id'] = torch.tensor(group_ids, dtype=torch.long)
        
        # Also store task_name and group_name as lists (for logging/debugging)
        if 'task_name' in examples[0]:
            batch['task_name'] = [e.get('task_name', '') for e in examples]
        if 'group_name' in examples[0]:
            batch['group_name'] = [e.get('group_name', '') for e in examples]
        
        return batch