import torch
import gc
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from models.heads import (
    MeanPooling, MaxPooling,
    ConvBERTClassificationHead, ConvBERTAttentionHead, ConvBERTMaxPoolingHead,
    MLPClassificationHead, BiLSTMClassificationHead
)


class FrozenPlmModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.layer_norm = nn.LayerNorm(args.hidden_size)

        if args.pooling_method == 'convbert':
            self.classifier = ConvBERTClassificationHead(args.hidden_size, args.num_labels, dropout=args.pooling_dropout)
        elif args.pooling_method == 'convbert_attention':
            self.classifier = ConvBERTAttentionHead(args.hidden_size, args.num_labels, dropout=args.pooling_dropout)
        elif args.pooling_method == 'convbert_max':
            self.classifier = ConvBERTMaxPoolingHead(args.hidden_size, args.num_labels, dropout=args.pooling_dropout)
        elif args.pooling_method == 'mlp':
            self.classifier = MLPClassificationHead(
                args.hidden_size, 
                args.num_labels, 
                hidden_dims=args.mlp_hidden_dims,
                dropout=args.pooling_dropout,
                activation=args.mlp_activation
            )
        elif args.pooling_method == 'bilstm':
            self.classifier = BiLSTMClassificationHead(
                args.hidden_size,
                args.num_labels,
                lstm_hidden_size=getattr(args, 'lstm_hidden_size', 256),
                lstm_num_layers=getattr(args, 'lstm_num_layers', 2),
                dropout=args.pooling_dropout
            )
        else:
            raise ValueError(f"classifier method {args.pooling_method} not supported")

    def plm_embedding(self, plm_model, aa_seq, attention_mask):
        with torch.no_grad():
            # For T5 models, use only the encoder for classification tasks
            if hasattr(plm_model, 'config') and plm_model.config.model_type == 't5':
                outputs = plm_model.encoder(input_ids=aa_seq, attention_mask=attention_mask)
                seq_embeds = outputs.last_hidden_state
            else:
                # For other models (BERT, ESM2, etc.)
                outputs = plm_model(input_ids=aa_seq, attention_mask=attention_mask)
                seq_embeds = outputs.last_hidden_state
            
        gc.collect()
        torch.cuda.empty_cache()
        return seq_embeds

    def forward(self, plm_model, batch):
        aa_seq, attention_mask = batch['aa_seq_input_ids'], batch['aa_seq_attention_mask']
        seq_embeds = self.plm_embedding(plm_model, aa_seq, attention_mask)

        # Use sequence embeddings directly for classification
        logits = self.classifier(seq_embeds, attention_mask)

        return logits


class FrozenFeatureExtractorModel(nn.Module):
    """
    Model that uses a pre-trained feature extractor (PLM + ConvBERT without decoder) as frozen feature extractor,
    and a new MLP classifier head for classification.
    The feature extractor model already contains PLM embedding and ConvBERT (without decoder).
    """
    def __init__(self, args, feature_extractor_model=None, feature_extractor_plm_model=None):
        super().__init__()
        self.args = args
        
        # Store feature extractor model (frozen) - this includes PLM + ConvBERT (without decoder)
        if feature_extractor_model is not None:
            self.feature_extractor_model = feature_extractor_model
            self.feature_extractor_plm_model = feature_extractor_plm_model
            # Freeze feature extractor
            for param in self.feature_extractor_model.parameters():
                param.requires_grad = False
            self.feature_extractor_model.eval()
            if self.feature_extractor_plm_model is not None:
                for param in self.feature_extractor_plm_model.parameters():
                    param.requires_grad = False
                self.feature_extractor_plm_model.eval()
        else:
            self.feature_extractor_model = None
            self.feature_extractor_plm_model = None
        
        self.layer_norm = nn.LayerNorm(args.hidden_size)

        # Create new classifier head (MLP)
        if args.pooling_method == 'mlp':
            self.classifier = MLPClassificationHead(
                args.hidden_size, 
                args.num_labels, 
                hidden_dims=getattr(args, 'mlp_hidden_dims', [512]),
                dropout=args.pooling_dropout,
                activation=getattr(args, 'mlp_activation', 'relu')
            )
        else:
            raise ValueError(f"For feature extractor model, only 'mlp' pooling_method is supported, got {args.pooling_method}")

    def forward(self, plm_model, batch):
        # Note: plm_model parameter is ignored when feature_extractor_model is provided
        # because the feature extractor already contains PLM
        
        # Apply feature extractor if provided (this includes PLM embedding + ConvBERT without decoder)
        if self.feature_extractor_model is not None:
            # The feature extractor model's forward already handles PLM embedding internally
            # Since decoder is removed, it returns features (ConvBERT output) directly
            with torch.no_grad():
                features = self.feature_extractor_model(self.feature_extractor_plm_model, batch)
        else:
            # Fallback: should not happen in normal usage
            raise ValueError("Feature extractor model is required for FrozenFeatureExtractorModel")

        # Apply layer norm
        features = self.layer_norm(features)

        # Use new classifier head for classification
        logits = self.classifier(features, batch.get('aa_seq_attention_mask'))

        return logits
