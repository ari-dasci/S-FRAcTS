import torch.nn as nn
import torch
from model.attn import AttentionLayer

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        attn_weights_list = []
        output = src
        for layer in self.layers:
            output, attn_weights = layer(output, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
            attn_weights_list.append(attn_weights)
        
        return output, attn_weights_list
    
class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer class for the Transformer model.

    This module represents a single layer of the Transformer encoder. It consists of
    multi-head self-attention, feedforward neural networks, and residual connections with
    layer normalization.

    Parameters
    ----------
    d_model : int
        The dimensionality of the model, representing the number of features or channels in
        the input and output embeddings.
    nhead : int
        The number of attention heads in the multi-head self-attention mechanism.
    attn : nn module for attention
        The attention module to use in the Transformer Encoder Layer.
    w_size : int
        The window size for the attention mechanism.
    dim_feedforward : int, optional
        The dimensionality of the feedforward neural networks, by default 16.
    dropout : float, optional
        The dropout probability applied to various parts of the layer, by default 0.
    l1 : nn module, optional
        The linear transformation module to use for the first part of the feedforward network,
        by default nn.Linear.
    l1_args : dict, optional
        The arguments to pass to the first linear transformation module, by default {"dtype":torch.float64}.
    l2 : nn module, optional  
        The linear transformation module to use for the second part of the feedforward network,
        by default nn.Linear.

    Attributes
    ----------
    self_attn : nn.MultiheadAttention
        Multi-head self-attention mechanism.
    l1 : nn module
        Linear transformation for the first part of the feedforward network.
    dropout : nn.Dropout
        Dropout layer applied to the input and output of the self-attention and feedforward networks.
    l2 : nn module
        Linear transformation for the second part of the feedforward network.
    dropout1 : nn.Dropout
        Dropout layer applied to the output of the first part of the feedforward network.
    dropout2 : nn.Dropout
        Dropout layer applied to the output of the second part of the feedforward network.
    activation : nn.LeakyReLU
        Activation function used in the feedforward network.

    Methods
    -------
    forward(src, src_mask=None, src_key_padding_mask=None, is_causal=None)
        Forward pass through the Transformer Encoder Layer.

    Examples
    --------
    Create a TransformerEncoderLayer instance:

    >>> encoder_layer = TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1)
    >>> input_data = torch.rand(32, 100, 512)
    >>> output_data = encoder_layer(input_data)

    Notes
    -----
    The Transformer Encoder Layer is a building block in the Transformer model, responsible for
    processing the input sequence and capturing dependencies between tokens.

    References
    ----------
    Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., ... & Polosukhin, I. (2017).
    Attention is all you need. In Advances in neural information processing systems (pp. 30-31).

    """
    def __init__(self, d_model, nhead, attn, attn_params, w_size, dim_feedforward=None, dropout=0, 
                 l1=nn.Linear, l1_args={"dtype":torch.float64}, 
                 l2=nn.Linear, l2_args={"dtype":torch.float64}):
        super(TransformerEncoderLayer, self).__init__()
        self.dim_feedforward = dim_feedforward if dim_feedforward is not None else d_model
        # attn parameter must be the class itself, not an instance
        self.attn = attn(**attn_params)
        self.self_attn = AttentionLayer(self.attn, w_size, d_model, nhead)
        self.l1 = l1(d_model, self.dim_feedforward, **l1_args)
        self.dropout = nn.Dropout(dropout)
        self.l2 = l2(self.dim_feedforward, d_model, **l2_args)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.normalization1 = nn.LayerNorm(d_model)
        self.normalization2 = nn.LayerNorm(d_model)

        self.activation = nn.LeakyReLU(True)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=None):
        """
        Apply forward pass through the Transformer Encoder Layer.

        Parameters
        ----------
        src : torch.Tensor
            The input tensor to the layer.
        src_mask : torch.Tensor, optional
            Mask to prevent attention to certain positions in the input sequence, by default None.
        src_key_padding_mask : torch.Tensor, optional
            Mask to prevent attention to padding positions in the input sequence, by default None.
        is_causal : bool, optional
            A flag to control the causality of the self-attention, by default None.

        Returns
        -------
        torch.Tensor
            The output tensor after processing through the layer.

        """
        
        src2, attn_weights = self.self_attn(src, src, src,attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.normalization1(src)
        residual_src = src
        src2 = self.l2(self.dropout(self.activation(self.l1(src))))
        src = src + self.dropout2(src2)
        # Add residual connection and layer normalization
        src = src + residual_src
        src = self.normalization2(src)
        return src,attn_weights