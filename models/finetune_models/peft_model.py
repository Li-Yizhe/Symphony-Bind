"""
use peft model
"""
import torch
import gc
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
try:
    from transformers import EsmTokenizer, EsmModel
except ImportError:
    # Fallback for older transformers versions
    from transformers import AutoTokenizer as EsmTokenizer, AutoModel as EsmModel
from peft import PeftModel, PeftConfig
from models.heads import (
    MeanPooling, MaxPooling,
    ConvBERTClassificationHead, ConvBERTAttentionHead, ConvBERTMaxPoolingHead,
    MLPClassificationHead, BiLSTMClassificationHead, BiLSTMPoolingHead,
    CNNClassificationHead, TransformerClassificationHead
)


class PEFTModel(nn.Module):
    """
    Parameter-Efficient Fine-Tuning model supporting multiple PEFT methods
    """

    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        if args.pooling_method == "convbert":
            self.classifier = ConvBERTClassificationHead(
                args.hidden_size, args.num_labels, dropout=args.pooling_dropout
            )
        elif args.pooling_method == "convbert_attention":
            self.classifier = ConvBERTAttentionHead(
                args.hidden_size, args.num_labels, dropout=args.pooling_dropout
            )
        elif args.pooling_method == "convbert_max":
            self.classifier = ConvBERTMaxPoolingHead(
                args.hidden_size, args.num_labels, dropout=args.pooling_dropout
            )
        elif args.pooling_method == "mlp":
            self.classifier = MLPClassificationHead(
                args.hidden_size, 
                args.num_labels, 
                hidden_dims=args.mlp_hidden_dims,
                dropout=args.pooling_dropout,
                activation=args.mlp_activation
            )
        elif args.pooling_method == "bilstm":
            self.classifier = BiLSTMClassificationHead(
                args.hidden_size,
                args.num_labels,
                lstm_hidden_size=getattr(args, 'lstm_hidden_size', 256),
                lstm_num_layers=getattr(args, 'lstm_num_layers', 2),
                dropout=args.pooling_dropout,
                bidirectional=getattr(args, 'lstm_bidirectional', True)
            )
        elif args.pooling_method == "bilstm_pooling":
            self.classifier = BiLSTMPoolingHead(
                args.hidden_size,
                args.num_labels,
                lstm_hidden_size=getattr(args, 'lstm_hidden_size', 256),
                lstm_num_layers=getattr(args, 'lstm_num_layers', 2),
                dropout=args.pooling_dropout,
                bidirectional=getattr(args, 'lstm_bidirectional', True),
                pooling=getattr(args, 'bilstm_pooling', 'mean')
            )
        elif args.pooling_method == "cnn":
            self.classifier = CNNClassificationHead(
                args.hidden_size,
                args.num_labels,
                nhead=getattr(args, 'nhead', 8),
                hidden_dim=getattr(args, 'hidden_dim', 512),
                num_hidden_layers=getattr(args, 'num_hidden_layers', 1),
                kernel_size=getattr(args, 'kernel_size', 7),
                dropout=args.pooling_dropout
            )
        elif args.pooling_method == "transformer":
            self.classifier = TransformerClassificationHead(
                args.hidden_size,
                args.num_labels,
                nhead=getattr(args, 'nhead', 8),
                hidden_dim=getattr(args, 'hidden_dim', 512),
                num_hidden_layers=getattr(args, 'num_hidden_layers', 1),
                kernel_size=getattr(args, 'kernel_size', 7),
                dropout=args.pooling_dropout
            )
        else:
            raise ValueError(f"classifier method {args.pooling_method} not supported")

    def plm_embedding(self, plm_model, aa_seq, attention_mask):
        # For T5 models, use only the encoder for classification tasks
        if hasattr(plm_model, 'encoder'):
            if (
                self.training
                and hasattr(self, "args")
                and self.args.training_method in ['plm-lora', 'plm-qlora', 'plm-dora', 'plm-adalora', 'plm-ia3']
            ):
                # For PEFT models, we need to access the base model's encoder
                if hasattr(plm_model, 'base_model'):
                    # For T5 models, the structure is different - base_model is the T5EncoderModel itself
                    if hasattr(plm_model.base_model, 'model'):
                        # ESM2 structure: use the full model forward pass
                        outputs = plm_model.base_model.model(aa_seq, attention_mask=attention_mask)
                    else:
                        # T5 structure: base_model.encoder (T5EncoderModel directly has encoder)
                        outputs = plm_model.base_model.encoder(input_ids=aa_seq, attention_mask=attention_mask)
                else:
                    # For ESM2, use the full model forward pass
                    outputs = plm_model(aa_seq, attention_mask=attention_mask)
            else:
                with torch.no_grad():
                    if hasattr(plm_model, 'base_model'):
                        # For T5 models, the structure is different - base_model is the T5EncoderModel itself
                        # For ESM2, use the full model forward pass
                        outputs = plm_model.base_model(aa_seq, attention_mask=attention_mask)
                    else:
                        # For ESM2, use the full model forward pass
                        outputs = plm_model(aa_seq, attention_mask=attention_mask)
        else:
            # For other models (BERT, ESM2, etc.)
            if (
                self.training
                and hasattr(self, "args")
                and self.args.training_method in ['plm-lora', 'plm-qlora', 'plm-dora', 'plm-adalora', 'plm-ia3']
            ):
                if "Prime" in self.args.plm_model:
                    outputs = plm_model(input_ids=aa_seq, attention_mask=attention_mask, output_hidden_states=True)
                else:
                    outputs = plm_model(input_ids=aa_seq, attention_mask=attention_mask)
            else:
                with torch.no_grad():
                    outputs = plm_model(input_ids=aa_seq, attention_mask=attention_mask)
        
        if "Prime" in self.args.plm_model:
            seq_embeds = outputs.sequence_hidden_states[-1]
        else:
            seq_embeds = outputs.last_hidden_state
        
        gc.collect()
        torch.cuda.empty_cache()
        return seq_embeds

    def forward(self, plm_model, batch):
        aa_seq, attention_mask = (
            batch["aa_seq_input_ids"],
            batch["aa_seq_attention_mask"],
        )
        seq_embeds = self.plm_embedding(plm_model, aa_seq, attention_mask)
        logits = self.classifier(seq_embeds, attention_mask)
        return logits