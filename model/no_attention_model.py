import torch
import torch.nn as nn
import math
from model.embed import PositionalEncoding
from model.base_module import Transformer_BaseModule


class FFNLayer(nn.Module):
    """Simple feedforward layer with residual connection and layer norm (no attention)."""
    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super(FFNLayer, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward, dtype=torch.float64)
        self.linear2 = nn.Linear(dim_feedforward, d_model, dtype=torch.float64)
        self.norm = nn.LayerNorm(d_model, dtype=torch.float64)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = self.norm(x + residual)
        return x


class NoAttentionModel(Transformer_BaseModule):
    """
    Same interface as TransformerModel but WITHOUT any attention layers.
    Only positional encoding + stacked FFN layers + output projection.
    Used as ablation baseline to measure the contribution of attention.
    """
    def __init__(self, feats, loss, optimizers, metrics,
                 d_model=None, n_head=8, dim_feedforward=None, d_model_out=None,
                 attn=None, attn_params=None, n_window=10, n_pred=None,
                 num_encoder_layers=1, num_decoder_layers=1, **kwargs):
        super(NoAttentionModel, self).__init__(loss, optimizers, metrics, **kwargs)
        self.name = 'NoAttention_FFN'
        self.n_feats = feats
        self.n_window = n_window
        self.n_pred = n_window if n_pred is None else n_pred
        self.d_model = d_model if d_model is not None else feats
        self.dim_feedforward = dim_feedforward if dim_feedforward is not None else 4 * feats
        self.d_model_out = d_model_out if d_model_out is not None else feats
        self.attn_params = attn_params if attn_params is not None else {"output_attention": True}

        # Positional encoding (same as original)
        self.pos_encoder = PositionalEncoding(self.d_model, 0.1, self.n_window)

        # Stacked FFN layers replacing encoder + decoder
        total_layers = num_encoder_layers + num_decoder_layers
        self.ffn_layers = nn.ModuleList([
            FFNLayer(self.d_model, self.dim_feedforward, dropout=0.1)
            for _ in range(total_layers)
        ])

        # Output projection: adjust time dimension and feature dimension
        self.proj_time = nn.Linear(self.n_window, self.n_pred, dtype=torch.float64)
        self.proj_feat = nn.Linear(self.d_model, self.d_model_out, dtype=torch.float64)

    def forward(self, src, tgt):
        # Scale + positional encoding (same as TransformerModel)
        x = src * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        # Pass through stacked FFN layers
        for layer in self.ffn_layers:
            x = layer(x)

        # Output projection: (batch, n_window, d_model) -> (batch, n_pred, d_model_out)
        x = x.permute(0, 2, 1)          # (batch, d_model, n_window)
        x = self.proj_time(x)           # (batch, d_model, n_pred)
        x = x.permute(0, 2, 1)          # (batch, n_pred, d_model)
        x = self.proj_feat(x)           # (batch, n_pred, d_model_out)

        # Return same format as TransformerModel for compatibility
        if self.attn_params.get("output_attention", False):
            return x, [], [], [], [], []
        else:
            return x
