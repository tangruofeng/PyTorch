import torch.nn as nn
from mha import MultiHeadAttention
from ln import LayerNorm
from ffl import FeedForwardLayer

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, head, forward_expansion, dropout):
        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadAttention(embed_size, head)
        self.norm1 = LayerNorm(embed_size)
        self.norm2 = LayerNorm(embed_size)
        self.feed_forward = FeedForwardLayer(embed_size, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask):
        attention = self.attn(query, key, value, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class Encoder(nn.Module):
    def __init__(
            self,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout=0.1,
    ):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, forward_expansion, dropout)
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, x, x, mask)
        return x