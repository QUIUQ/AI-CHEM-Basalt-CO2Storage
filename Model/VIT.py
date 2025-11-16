from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class LearnedPositionalEmbeddings(nn.Module):

    def __init__(self, d_model: int, max_len: int = 50 ):
        """
        * `d_model` is the transformer embeddings size
        * `max_len` is the maximum number of patches
        """
        super().__init__()
        # Positional embeddings for each location
        self.positional_encodings = nn.Parameter(torch.zeros(1 ,max_len, d_model), requires_grad=True)

    def forward(self, x: torch.Tensor):
        """
        * `x` is the patch embeddings of shape `[ batch_size, patches , d_model]`
        """
        # Get the positional embeddings for the given patches
        pe = self.positional_encodings[:,:x.shape[1],:]
        # Add to patch embeddings and return
        return x + pe


class SpatialTransformer (nn.Module):
    """
    ## Spatial Transformer
    """

    def __init__(self ,channels: int ,n_heads: int ,n_layers: int ):
        """
        :param channels: is the number of channels in the feature map
        :param n_heads: is the number of attention heads
        :param n_layers: is the number of transformer layers
        :param d_cond: is the size of the conditional embedding
        """
        super ().__init__ ()
        # Initial group normalization
        self.norm = torch.nn.GroupNorm (num_groups=16 ,num_channels=channels ,eps=1e-6 ,affine=True)
        # Initial $1 \times 1$ convolution
        self.proj_in = nn.Conv2d (channels ,channels ,kernel_size=1 ,stride=1 ,padding=0)

        self.pos_emb = LearnedPositionalEmbeddings(channels)

        # Transformer layers
        self.transformer_blocks = nn.ModuleList (
            [BasicTransformerBlock (channels ,n_heads , 32 ) for _ in range (n_layers)]
        )

        # Final $1 \times 1$ convolution
        self.proj_out = nn.Conv2d (channels ,channels ,kernel_size=1 ,stride=1 ,padding=0)

    def forward(self ,x: torch.Tensor ):
        """
        :param x: is the feature map of shape `[batch_size, channels, height, width]`
        :param cond: is the conditional embeddings of shape `[batch_size,  n_cond, d_cond]`
        """
        # Get shape `[batch_size, channels, height, width]`
        b ,c ,h ,w = x.shape
        # For residual connection
        x_in = x
        # Normalize
        x = self.norm (x)
        # Initial $1 \times 1$ convolution
        x = self.proj_in (x)
        # Transpose and reshape from `[batch_size, channels, height, width]`
        # to `[batch_size, height * width, channels]`
        x = x.permute (0 ,2 ,3 ,1).view (b ,h * w ,c)
        x= self.pos_emb(x)
        # Apply the transformer layers
        for block in self.transformer_blocks:
            x = block (x)
        # Reshape and transpose from `[batch_size, height * width, channels]`
        # to `[batch_size, channels, height, width]`
        x = x.view (b ,h ,w ,c).permute (0 ,3 ,1 ,2)
        # Final $1 \times 1$ convolution
        x = self.proj_out (x)
        # Add residual
        return x + x_in


class BasicTransformerBlock (nn.Module):
    """
    ### Transformer Layer
    """

    def __init__(self ,d_model: int ,n_heads: int ,d_head: int):
        """
        :param d_model: is the input embedding size
        :param n_heads: is the number of attention heads
        :param d_head: is the size of a attention head
        """
        super ().__init__ ()
        # Self-attention layer and pre-norm layer
        self.attn1 = CrossAttention (d_model ,d_model ,n_heads ,d_head)
        self.norm1 = nn.LayerNorm (d_model)
        # Feed-forward network and pre-norm layer
        self.ff = FeedForward (d_model)
        self.norm3 = nn.LayerNorm (d_model)

    def forward(self ,x: torch.Tensor ):
        """
        :param x: are the input embeddings of shape `[batch_size, height * width, d_model]`
        :param cond: is the conditional embeddings of shape `[batch_size,  n_cond, d_cond]`
        """
        # Self attention
        x = self.attn1 (self.norm1 (x)) + x

        # Feed-forward network
        x = self.ff (self.norm3 (x)) + x
        #
        return x


class CrossAttention (nn.Module):
    """
    ### Cross Attention Layer

    This falls-back to self-attention when conditional embeddings are not specified.
    """

    use_flash_attention: bool = False

    def __init__(self ,d_model: int ,d_cond: int ,n_heads: int ,d_head: int ,is_inplace: bool = True):
        """
        :param d_model: is the input embedding size
        :param n_heads: is the number of attention heads
        :param d_head: is the size of a attention head
        :param d_cond: is the size of the conditional embeddings
        :param is_inplace: specifies whether to perform the attention softmax computation inplace to
            save memory
        """
        super ().__init__ ()

        self.is_inplace = is_inplace
        self.n_heads = n_heads
        self.d_head = d_head


        # Attention scaling factor
        self.scale = d_head ** -0.5

        # Query, key and value mappings
        d_attn = d_head * n_heads
        self.to_q = nn.Linear (d_model ,d_attn ,bias=False)
        self.to_k = nn.Linear (d_cond ,d_attn ,bias=False)
        self.to_v = nn.Linear (d_cond ,d_attn ,bias=False)


        self.to_out = nn.Sequential (nn.Linear (d_attn ,d_model))

        try:

            from flash_attn.flash_attention import FlashAttention
            self.flash = FlashAttention ()

            self.flash.softmax_scale = self.scale

        except ImportError:
            self.flash = None

    def forward(self ,x: torch.Tensor ,cond: Optional [torch.Tensor] = None):
        """
        :param x: are the input embeddings of shape `[batch_size, height * width, d_model]`
        :param cond: is the conditional embeddings of shape `[batch_size, n_cond, d_cond]`
        """

        # If `cond` is `None` we perform self attention
        has_cond = cond is not None
        if not has_cond:
            cond = x

        # Get query, key and value vectors
        q = self.to_q (x)
        k = self.to_k (cond)
        v = self.to_v (cond)

        # Use flash attention if it's available and the head size is less than or equal to `128`
        if CrossAttention.use_flash_attention and self.flash is not None and not has_cond and self.d_head <= 128:
            return self.flash_attention (q ,k ,v)
        # Otherwise, fallback to normal attention
        else:
            return self.normal_attention (q ,k ,v)

    def flash_attention(self ,q: torch.Tensor ,k: torch.Tensor ,v: torch.Tensor):
        """
        #### Flash Attention

        :param q: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        :param k: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        :param v: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        """

        # Get batch size and number of elements along sequence axis (`width * height`)
        batch_size ,seq_len ,_ = q.shape

        # Stack `q`, `k`, `v` vectors for flash attention, to get a single tensor of
        # shape `[batch_size, seq_len, 3, n_heads * d_head]`
        qkv = torch.stack ((q ,k ,v) ,dim=2)
        # Split the heads
        qkv = qkv.view (batch_size ,seq_len ,3 ,self.n_heads ,self.d_head)

        # Flash attention works for head sizes `32`, `64` and `128`, so we have to pad the heads to
        # fit this size.
        if self.d_head <= 32:
            pad = 32 - self.d_head
        elif self.d_head <= 64:
            pad = 64 - self.d_head
        elif self.d_head <= 128:
            pad = 128 - self.d_head
        else:
            raise ValueError (f'Head size ${self.d_head} too large for Flash Attention')

        # Pad the heads
        if pad:
            qkv = torch.cat ((qkv ,qkv.new_zeros (batch_size ,seq_len ,3 ,self.n_heads ,pad)) ,dim=-1)

        # Compute attention
        # $$\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_{key}}}\Bigg)V$$
        # This gives a tensor of shape `[batch_size, seq_len, n_heads, d_padded]`
        out ,_ = self.flash (qkv)
        # Truncate the extra head size
        out = out [: ,: ,: ,:self.d_head]
        # Reshape to `[batch_size, seq_len, n_heads * d_head]`
        out = out.reshape (batch_size ,seq_len ,self.n_heads * self.d_head)

        # Map to `[batch_size, height * width, d_model]` with a linear layer
        return self.to_out (out)

    def normal_attention(self ,q: torch.Tensor ,k: torch.Tensor ,v: torch.Tensor):
        """
        #### Normal Attention

        :param q: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        :param k: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        :param v: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        """

        # Split them to heads of shape `[batch_size, seq_len, n_heads, d_head]`
        q = q.view (*q.shape [:2] ,self.n_heads ,-1)
        #print(q.shape)
        k = k.view (*k.shape [:2] ,self.n_heads ,-1)
        v = v.view (*v.shape [:2] ,self.n_heads ,-1)


        attn = torch.einsum ('bihd,bjhd->bhij' ,q ,k) * self.scale


        if self.is_inplace:
            half = attn.shape [0] // 2
            attn [half:] = attn [half:].softmax (dim=-1)
            attn [:half] = attn [:half].softmax (dim=-1)
        else:
            attn = attn.softmax (dim=-1)

        # Compute attention output
        out = torch.einsum ('bhij,bjhd->bihd' ,attn ,v)
        # Reshape to `[batch_size, height * width, n_heads * d_head]`
        out = out.reshape (*out.shape [:2] ,-1)
        # Map to `[batch_size, height * width, d_model]` with a linear layer
        return self.to_out (out)


class FeedForward (nn.Module):
    """
    ### Feed-Forward Network
    """

    def __init__(self ,d_model: int ,d_mult: int = 4):
        """
        :param d_model: is the input embedding size
        :param d_mult: is multiplicative factor for the hidden layer size
        """
        super ().__init__ ()
        self.net = nn.Sequential (
            GeGLU (d_model ,d_model * d_mult) ,
            nn.Dropout (0.1) ,
            nn.Linear (d_model * d_mult ,d_model)
        )

    def forward(self ,x: torch.Tensor):
        return self.net (x)


class GeGLU (nn.Module):

    def __init__(self ,d_in: int ,d_out: int):
        super ().__init__ ()
        self.proj = nn.Linear (d_in ,d_out * 2)

    def forward(self ,x: torch.Tensor):

        x ,gate = self.proj (x).chunk (2 ,dim=-1)
        return x * F.gelu (gate)