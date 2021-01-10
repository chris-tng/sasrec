from flax import linen as nn
from jax import numpy as jnp


class FeedForward(nn.Module):
    """Position-wise FeedForward with Gates Linear Units and GELU activation
    as in https://arxiv.org/pdf/2002.05202.pdf
    """
    multiplicative: int = 4  # d_ff / d_model
    dropout_rate: float = 0.

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        d_model = x.shape[-1]
        # keep similar number of params wrt non-GLU
        mult = int(self.multiplicative/3 * 2)
        d_ff = d_model * mult

        gate = nn.Dense(d_ff, use_bias=False, name="wi_0")(x)
        x = nn.Dense(d_ff, use_bias=False, name="wi_1")(x)
        x = nn.gelu(gate, approximate=True) * x
        x = nn.Dropout(rate=self.dropout_rate, name="dropout")(x, deterministic=deterministic)
        x = nn.Dense(d_model, use_bias=False, name="wo")(x)
        return x


class SelfAttention(nn.Module):
    num_heads: int
    causal: bool = True
    dropout_rate: float = 0.0
    
    @nn.compact
    def __call__(self, x, attn_mask, deterministic: bool = True):
        """
        Args:
        - x of shape (b, n, d)
        - attn_mask of shape (b, n)
        """
        b, n, d, h = *x.shape, self.num_heads
        head_size = d / h

        # (b, n, d*3)
        x = nn.Dense(d * 3, use_bias=False, name="qkv_projection")(x)
        
        # (b, n, d) -> (b, n, h, hsize) -> (b, h, n, hsize)
        q, k, v = [_x.reshape(b, n, h, head_size).transpose((0, 2, 1, 3)) for _x in x.split(3, axis=-1)] 
        # attention : (b, h, n, n)
        attention = q @ k.transpose((0,1,3,2)) * (d ** -0.5)
        
        # Fill -inf into mask
        if attn_mask is not None:
            # (b, 1, n, 1) * (b, 1, 1, n)
            # TODO: need to check
            attn_mask = attn_mask[:, None, :, None] * attn_mask[:, None, None, :]
        else:
            if self.causal:
                attn_mask = jnp.triu( jnp.ones( (n, n) ), k=1 )
                attn_mask = 1. - attn_mask[None, None, :, :]
                
        if attn_mask is not None:
            attention = attention * attn_mask
            attention = jnp.where(attention == 0., -jnp.inf, attention)
        
        attention_weights = nn.softmax(attention, axis=-1)
        attention_weights = nn.Dropout(self.dropout_rate)(attention_weights, deterministic=deterministic)
        # context : (b, h, n, hsize) = (b, h, n, n) * (b, h, n, hsize) 
        context = attention_weights @ v
        context = context.transpose((0, 2, 1, 3)).reshape((b,n,d))
        context = nn.Dense(d, name="out_projection")(context)
        return context


class SubLayer(nn.Module):
    num_heads: int
    ff_multiplicative: int = 4
    causal: bool = False
    attention_dropout: float = 0.
    ff_dropout: float = 0.

    def setup(self):
        self.ln0 = nn.LayerNorm()
        self.self_attn = SelfAttention(num_heads=self.num_heads, causal=self.causal, dropout_rate=self.attention_dropout)
        self.ln1 = nn.LayerNorm()
        self.ffn = FeedForward(multiplicative=self.ff_multiplicative, dropout_rate=self.ff_dropout)

    def __call__(self, x, attention_mask = None, deterministic = False):
        x = self.self_attn(self.ln0(x), attention_mask, deterministic=deterministic) + x
        x = self.ffn(self.ln1(x), deterministic=deterministic) + x
        return x


class Encoder(nn.Module):
    N: int
    num_heads: int
    causal: bool = False
    ff_multiplicative: int = 4
    attention_dropout: float = 0.
    ff_dropout: float = 0.

    def setup(self):
        self.layers = [
            SubLayer(num_heads=self.num_heads, 
                     causal=self.causal,
                     ff_multiplicative=self.ff_multiplicative, 
                     attention_dropout=self.attention_dropout, 
                     ff_dropout=self.ff_dropout)
            for i in range(self.N)
        ]

    def __call__(self, x, src_mask = None, deterministic = False):        
        for layer in self.layers:
            x = layer(x, attention_mask = src_mask, deterministic=deterministic)

        return x


class Transformer(nn.Module):
    num_tokens: int
    embed_size: int
    max_seq_len: int
    N: int
    num_heads: int
    causal: bool = False
    ff_multiplicative: int = 4
    attention_dropout: float = 0.
    ff_dropout: float = 0.
    
    def setup(self):
        self.token_emb = nn.Embed(num_embeddings=self.num_tokens, features=self.embed_size)
        self.pos_emb = nn.Embed(num_embeddings=self.max_seq_len, features=self.embed_size)
        self.encoder = Encoder(N=self.N, num_heads=self.num_heads, causal=self.causal,
                               ff_multiplicative=self.ff_multiplicative,
                               attention_dropout=self.attention_dropout,
                               ff_dropout=self.ff_dropout)
        
    def __call__(self, x, deterministic: bool = False, *args, **kwargs):
        """
        - x : shape (b, n)
        """
        batch_size, seq_len = x.shape
        x = self.token_emb(x)  # (b, n, d)
        x = x + self.pos_emb( jnp.arange(seq_len) )[None, :]      
        # use default causal mask
        x = self.encoder(x, src_mask=None, deterministic=deterministic, *args, **kwargs)
        # x @ embedding.T : (b, n, d) @ (d, V) = (b, n, V)
        x = self.token_emb.attend(x)  # logits
        return x




