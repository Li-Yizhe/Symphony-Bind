import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from functools import partial
from transformers.models import convbert
from typing import Optional


class GlobalMaxPooling1D(nn.Module):
    def __init__(self):
        """Applies global max pooling over timesteps dimension"""
        super().__init__()
        self.global_max_pool1d = partial(torch.max, dim=1)

    def forward(self, x):
        out, _ = self.global_max_pool1d(x)
        return out


class GlobalAvgPooling1D(nn.Module):
    def __init__(self):
        """Applies global average pooling over timesteps dimension"""
        super().__init__()
        self.global_avg_pool1d = partial(torch.mean, dim=1)

    def forward(self, x):
        out = self.global_avg_pool1d(x)
        return out


class ConvBERT(nn.Module):
    def __init__(
        self,
        input_dim: int,
        nhead: int,
        hidden_dim: int,
        num_hidden_layers: int = 1,
        kernel_size: int = 7,
        dropout: float = 0.2,
        pooling: str = None,
    ):
        """
        Base model that consists of ConvBert layer.
        """
        super().__init__()

        self.model_type = "Transformer"
        encoder_layers_Config = convbert.ConvBertConfig(
            hidden_size=input_dim,
            num_attention_heads=nhead,
            intermediate_size=hidden_dim,
            conv_kernel_size=kernel_size,
            hidden_dropout_prob=dropout,
            num_hidden_layers=num_hidden_layers,
        )

        self.encoder = convbert.ConvBertModel(encoder_layers_Config).encoder

        if pooling is not None:
            if pooling in {"avg", "mean"}:
                self.pooling = GlobalAvgPooling1D()
            elif pooling == "max":
                self.pooling = GlobalMaxPooling1D()
            else:
                raise ValueError(
                    "Expected pooling to be [`avg`, `max`]. "
                    f"Received: {pooling}"
                )

    def get_extended_attention_mask(
        self,
        attention_mask: torch.LongTensor,
    ) -> torch.Tensor:
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (
            1.0 - extended_attention_mask
        ) * torch.finfo(torch.float32).min
        return extended_attention_mask

    def forward(
        self,
        x: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        attention_mask = self.get_extended_attention_mask(attention_mask)
        x = self.encoder(x, attention_mask=attention_mask)[0]
        if hasattr(self, 'pooling'):
            x = self.pooling(x)
        return x


class MaskedConv1d(nn.Conv1d):
    """Masked 1D convolution layer"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        padding = dilation * (kernel_size - 1) // 2
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding=padding,
        )

    def forward(self, x, input_mask=None):
        if input_mask is not None:
            x = x * input_mask.unsqueeze(-1)
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class Attention1dPooling(nn.Module):
    """Attention pooling method"""
    def __init__(self, hidden_size):
        super().__init__()
        self.layer = MaskedConv1d(hidden_size, 1, 1)

    def forward(self, x, input_mask=None):
        batch_size = x.shape[0]
        attn = self.layer(x)
        attn = attn.view(batch_size, -1)
        
        if input_mask is not None:
            attn = attn.masked_fill_(
                ~input_mask.view(batch_size, -1).bool(), float("-inf")
            )
        
        attn = F.softmax(attn, dim=-1).view(batch_size, -1, 1)
        out = (attn * x).sum(dim=1)
        return out


class MeanPooling(nn.Module):
    """Mean pooling method"""
    def __init__(self):
        super().__init__()

    def forward(self, features, input_mask=None):
        if input_mask is not None:
            masked_features = features * input_mask.unsqueeze(2)
            sum_features = torch.sum(masked_features, dim=1)
            mean_pooled_features = sum_features / input_mask.sum(dim=1, keepdim=True)
        else:
            mean_pooled_features = torch.mean(features, dim=1)
        return mean_pooled_features


class MaxPooling(nn.Module):
    """Max pooling method"""
    def __init__(self):
        super().__init__()

    def forward(self, features, input_mask=None):
        if input_mask is not None:
            masked_features = features * input_mask.unsqueeze(2)
            masked_features = masked_features.masked_fill(
                ~input_mask.unsqueeze(-1).bool(), float("-inf")
            )
            max_pooled_features, _ = torch.max(masked_features, dim=1)
        else:
            max_pooled_features, _ = torch.max(features, dim=1)
        return max_pooled_features


class ConvBERTClassificationHead(nn.Module):
    """ConvBERT + Mean pooling classification head"""
    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        nhead: int = 8,
        hidden_dim: int = 512,
        num_hidden_layers: int = 1,
        kernel_size: int = 7,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.convbert = ConvBERT(
            input_dim=hidden_size,
            nhead=nhead,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            pooling=None,
        )
        
        self.num_labels = num_labels
        self.decoder = nn.Linear(hidden_size, num_labels)
        self.reset_parameters()

    def reset_parameters(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, input_mask=None):
        hidden_outputs = self.convbert(x, input_mask)
        
        batch_size, seq_len, hidden_dim = hidden_outputs.shape
        reshaped_outputs = hidden_outputs.view(-1, hidden_dim)
        logits = self.decoder(reshaped_outputs)
        logits = logits.view(batch_size, seq_len, 2)
        
        return logits


class ConvBERTAttentionHead(nn.Module):
    """ConvBERT + Attention pooling classification head"""
    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        nhead: int = 8,
        hidden_dim: int = 512,
        num_hidden_layers: int = 1,
        kernel_size: int = 7,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.convbert = ConvBERT(
            input_dim=hidden_size,
            nhead=nhead,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            pooling=None,
        )
        
        self.attention = nn.Linear(hidden_dim, 1)
        self.num_labels = num_labels
        self.decoder = nn.Linear(hidden_dim, num_labels)
        self.reset_parameters()

    def reset_parameters(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, input_mask=None):
        hidden_outputs = self.convbert(x, input_mask)
        
        attention_weights = self.attention(hidden_outputs)
        
        if input_mask is not None:
            attention_weights = attention_weights.masked_fill(
                ~input_mask.unsqueeze(-1).bool(), float("-inf")
            )
        
        attention_weights = F.softmax(attention_weights, dim=1)
        pooled_output = torch.sum(hidden_outputs * attention_weights, dim=1)
        
        logits = self.decoder(pooled_output)
        return logits


class ConvBERTMaxPoolingHead(nn.Module):
    """ConvBERT + Max pooling classification head"""
    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        nhead: int = 8,
        hidden_dim: int = 512,
        num_hidden_layers: int = 1,
        kernel_size: int = 7,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.convbert = ConvBERT(
            input_dim=hidden_size,
            nhead=nhead,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            pooling=None,
        )
        
        self.num_labels = num_labels
        self.decoder = nn.Linear(hidden_dim, num_labels)
        self.reset_parameters()

    def reset_parameters(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, input_mask=None):
        hidden_outputs = self.convbert(x, input_mask)
        
        if input_mask is not None:
            masked_outputs = hidden_outputs * input_mask.unsqueeze(-1)
            masked_outputs = masked_outputs.masked_fill(
                ~input_mask.unsqueeze(-1).bool(), float("-inf")
            )
            pooled_output, _ = torch.max(masked_outputs, dim=1)
        else:
            pooled_output, _ = torch.max(hidden_outputs, dim=1)
        
        logits = self.decoder(pooled_output)
        return logits


class MLPClassificationHead(nn.Module):
    """MLP classification head with 2 layers (1 hidden layer)"""
    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        hidden_dims: list = [512, 256],
        dropout: float = 0.2,
        activation: str = 'relu',
    ):
        super().__init__()
        
        self.num_labels = num_labels
        
        # Build 2-layer MLP: input -> hidden -> output
        # Use first hidden dimension from hidden_dims
        hidden_dim = hidden_dims[0] if hidden_dims else 512
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),  # Input layer -> Hidden layer
            nn.Dropout(dropout),
            nn.ReLU() if activation == 'relu' else nn.GELU(),
            nn.Linear(hidden_dim, num_labels)    # Hidden layer -> Output layer
        )
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters"""
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x, input_mask=None):
        """
        Forward pass for 2-layer MLP classification head
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size]
            input_mask: Optional attention mask of shape [batch_size, seq_len]
        Returns:
            logits: Output tensor of shape [batch_size, seq_len, num_labels]
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # Reshape to [batch_size * seq_len, hidden_size] for processing
        reshaped_x = x.view(-1, hidden_size)
        
        # Apply 2-layer MLP
        logits = self.mlp(reshaped_x)
        
        # Reshape back to [batch_size, seq_len, num_labels]
        logits = logits.view(batch_size, seq_len, self.num_labels)
        
        return logits


class BiLSTMClassificationHead(nn.Module):
    """BiLSTM classification head for sequence labeling"""
    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ):
        super().__init__()
        
        self.num_labels = num_labels
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.bidirectional = bidirectional
        
        # BiLSTM layer
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate output size based on bidirectional setting
        lstm_output_size = lstm_hidden_size * 2 if bidirectional else lstm_hidden_size
        
        # Classification layer
        self.decoder = nn.Linear(lstm_output_size, num_labels)
        self.dropout = nn.Dropout(dropout)
        
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters"""
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        # Initialize decoder weights
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x, input_mask=None):
        """
        Forward pass for BiLSTM classification head
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size]
            input_mask: Optional attention mask of shape [batch_size, seq_len]
        Returns:
            logits: Output tensor of shape [batch_size, seq_len, num_labels]
        """
        # LSTM forward pass
        lstm_output, _ = self.lstm(x)
        
        # Apply dropout
        lstm_output = self.dropout(lstm_output)
        
        # Apply classification layer
        logits = self.decoder(lstm_output)
        
        return logits


class BiLSTMPoolingHead(nn.Module):
    """BiLSTM with pooling classification head for sequence classification"""
    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
        pooling: str = 'mean',  # 'mean', 'max', 'attention'
    ):
        super().__init__()
        
        self.num_labels = num_labels
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.bidirectional = bidirectional
        self.pooling = pooling
        
        # BiLSTM layer
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate output size based on bidirectional setting
        lstm_output_size = lstm_hidden_size * 2 if bidirectional else lstm_hidden_size
        
        # Pooling layer
        if pooling == 'attention':
            self.attention = nn.Linear(lstm_output_size, 1)
        elif pooling in ['mean', 'max']:
            self.attention = None
        else:
            raise ValueError(f"Unsupported pooling method: {pooling}")
        
        # Classification layer
        self.decoder = nn.Linear(lstm_output_size, num_labels)
        self.dropout = nn.Dropout(dropout)
        
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters"""
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        # Initialize attention weights if using attention pooling
        if self.attention is not None:
            nn.init.xavier_uniform_(self.attention.weight)
            nn.init.zeros_(self.attention.bias)
        
        # Initialize decoder weights
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x, input_mask=None):
        """
        Forward pass for BiLSTM pooling classification head
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size]
            input_mask: Optional attention mask of shape [batch_size, seq_len]
        Returns:
            logits: Output tensor of shape [batch_size, num_labels]
        """
        # LSTM forward pass
        lstm_output, _ = self.lstm(x)
        
        # Apply dropout
        lstm_output = self.dropout(lstm_output)
        
        # Apply pooling
        if self.pooling == 'mean':
            if input_mask is not None:
                masked_outputs = lstm_output * input_mask.unsqueeze(-1)
                pooled_output = masked_outputs.sum(dim=1) / input_mask.sum(dim=1, keepdim=True)
            else:
                pooled_output = lstm_output.mean(dim=1)
        
        elif self.pooling == 'max':
            if input_mask is not None:
                masked_outputs = lstm_output * input_mask.unsqueeze(-1)
                masked_outputs = masked_outputs.masked_fill(
                    ~input_mask.unsqueeze(-1).bool(), float("-inf")
                )
                pooled_output, _ = torch.max(masked_outputs, dim=1)
            else:
                pooled_output, _ = torch.max(lstm_output, dim=1)
        
        elif self.pooling == 'attention':
            attention_weights = self.attention(lstm_output)
            
            if input_mask is not None:
                attention_weights = attention_weights.masked_fill(
                    ~input_mask.unsqueeze(-1).bool(), float("-inf")
                )
            
            attention_weights = F.softmax(attention_weights, dim=1)
            pooled_output = torch.sum(lstm_output * attention_weights, dim=1)
        
        # Apply classification layer
        logits = self.decoder(pooled_output)
        
        return logits


class ConvBertConvOnlyLayer(nn.Module):
    """
    CNN-only layer with STATIC convolution (no Query modulation)
    Uses SeparableConv1D but without dynamic kernel generation
    """
    def __init__(self, config):
        super().__init__()
        from transformers.models.convbert.modeling_convbert import SeparableConv1D
        
        self.conv_kernel_size = config.conv_kernel_size
        self.hidden_size = config.hidden_size
        
        # Static convolution components (no Query needed)
        self.conv_layer = SeparableConv1D(
            config, config.hidden_size, config.hidden_size, self.conv_kernel_size
        )
        
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states, attention_mask=None):
        # Apply static separable convolution
        conv_output = self.conv_layer(hidden_states.transpose(1, 2))
        conv_output = conv_output.transpose(1, 2)
        
        # Apply dropout and layer norm with residual
        conv_output = self.dropout(conv_output)
        output = self.layer_norm(conv_output + hidden_states)
        
        return output


class ConvBertAttentionOnlyLayer(nn.Module):
    """
    ConvBERT layer with ONLY standard self-attention (no convolution)
    Standard Q, K, V attention without SeparableConv1D
    """
    def __init__(self, config):
        super().__init__()
        
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number "
                f"of attention heads ({config.num_attention_heads})"
            )
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Standard Q, K, V (no convolution)
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, attention_mask=None):
        # Standard self-attention (Q, K, V)
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / torch.sqrt(
            torch.tensor(self.attention_head_size, dtype=torch.float32, device=attention_scores.device)
        )
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Normalize attention scores to probabilities
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # Output projection
        attention_output = self.dense(context_layer)
        attention_output = self.dropout(attention_output)
        
        # Layer norm with residual
        output = self.layer_norm(attention_output + hidden_states)
        
        return output


class CNNClassificationHead(nn.Module):
    """
    CNN-only classification head for STRICT ablation study
    Uses STATIC SeparableConv1D (no Query modulation, no dynamic convolution)
    This is pure convolution without any attention components
    Uses same parameter design as original ConvBERTClassificationHead
    """
    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        nhead: int = 8,
        hidden_dim: int = 512,
        num_hidden_layers: int = 1,
        kernel_size: int = 7,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.num_labels = num_labels
        
        # Create ConvBERT config - same as original ConvBERT
        config = convbert.ConvBertConfig(
            hidden_size=hidden_size,  # This is the input_dim from ESM2 (1280)
            num_attention_heads=nhead,
            intermediate_size=hidden_dim,
            conv_kernel_size=kernel_size,
            hidden_dropout_prob=dropout,
            num_hidden_layers=num_hidden_layers,
        )
        
        # Build layers with ONLY SeparableConv1D (static convolution)
        self.layers = nn.ModuleList([
            ConvBertConvOnlyLayer(config) 
            for _ in range(num_hidden_layers)
        ])
        
        self.decoder = nn.Linear(hidden_size, num_labels)
        self.reset_parameters()

    def reset_parameters(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, input_mask=None):
        batch_size, seq_len, hidden_dim = x.shape
        
        # Apply STATIC SeparableConv1D layers (no Query modulation)
        hidden_states = x
        for layer in self.layers:
            hidden_states = layer(hidden_states, input_mask)
        
        # Classification head
        reshaped_outputs = hidden_states.view(-1, hidden_dim)
        logits = self.decoder(reshaped_outputs)
        logits = logits.view(batch_size, seq_len, self.num_labels)
        
        return logits


class TransformerClassificationHead(nn.Module):
    """
    Transformer-only classification head for STRICT ablation study
    Uses ONLY standard Q, K, V self-attention (NO convolution)
    This is pure multi-head self-attention without ConvBERT's SeparableConv1D
    """
    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        nhead: int = 8,
        hidden_dim: int = 512,
        num_hidden_layers: int = 1,
        kernel_size: int = 7,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.num_labels = num_labels
        
        # Create ConvBERT config - same as original ConvBERT
        config = convbert.ConvBertConfig(
            hidden_size=hidden_size,  # This is the input_dim from ESM2 (1280)
            num_attention_heads=nhead,
            intermediate_size=hidden_dim,
            conv_kernel_size=kernel_size,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            num_hidden_layers=num_hidden_layers,
        )
        
        # Build layers with ONLY standard self-attention (no convolution)
        self.layers = nn.ModuleList([
            ConvBertAttentionOnlyLayer(config) 
            for _ in range(num_hidden_layers)
        ])
        
        self.decoder = nn.Linear(hidden_size, num_labels)
        self.reset_parameters()

    def reset_parameters(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def get_extended_attention_mask(self, attention_mask: torch.LongTensor) -> torch.Tensor:
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(torch.float32).min
        return extended_attention_mask

    def forward(self, x, input_mask=None):
        batch_size, seq_len, hidden_dim = x.shape
        
        # Prepare attention mask
        if input_mask is not None:
            attention_mask = self.get_extended_attention_mask(input_mask)
        else:
            attention_mask = None
        
        # Apply standard self-attention layers (NO convolution)
        hidden_states = x
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Classification head
        reshaped_outputs = hidden_states.view(-1, hidden_dim)
        logits = self.decoder(reshaped_outputs)
        logits = logits.view(batch_size, seq_len, self.num_labels)
        
        return logits