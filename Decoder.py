import torch.nn as nn
from mha import MultiHeadAttention
from ln import LayerNorm
from encoder import TransformerBlock

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout=0.1):
        super(DecoderBlock, self).__init__()
        self.norm = LayerNorm(embed_size)
        self.attn = MultiHeadAttention(embed_size, heads, dropout)
        self.transformer = TransformerBlock(embed_size, heads, forward_expansion, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attn = self.attn(x, x, x, trg_mask)
        query = self.dropout(self.norm(attn+x))
        out = self.attn(query, value, key, src_mask)
        return out

class Decoder(nn.Module):
    def __init__(
            self,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout=0.1,
    ):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout)
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_out, src_mask, trg_mask):
        for layer in self.layers:
            x = layer(x, encoder_out, encoder_out, src_mask, trg_mask)
        return x