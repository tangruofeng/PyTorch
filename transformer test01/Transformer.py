import torch.nn as nn
import torch
from encoder import Encoder
from Decoder import Decoder

class Transformer(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            trg_vocab_size,
            src_pad_idx,
            trg_pad_idx,
            embed_size=512,
            num_encoder_layers=6,
            num_decoder_layers=6,
            forward_expansion=4,
            heads=8,
            dropout=0,
            max_length=100,
            device="cpu",
    ):
        super(Transformer, self).__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

        self.encoder = Encoder(
            embed_size,
            num_encoder_layers,
            heads,
            forward_expansion,
            dropout
        )
        self.decoder = Decoder(
            embed_size,
            num_decoder_layers,
            heads,
            forward_expansion,
            dropout
        )
        self.src_word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.src_position_embedding = nn.Embedding(max_length, embed_size)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.trg_position_embedding = nn.Embedding(max_length, embed_size)

        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, trg_len, trg_len
        )

    def forward(self, src, trg):
        N, src_seq_length = src.shape
        N, trg_seq_length = trg.shape
        src_positions = (
            torch.arange(0, src_seq_length)
            .unsqueeze(0)
            .expand(N, src_seq_length)
            .to(self.device)
        )
        trg_positions = (
            torch.arange(0, trg_seq_length)
                .unsqueeze(0)
                .expand(N, trg_seq_length)
                .to(self.device)
        )
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        x = self.dropout(
            self.src_word_embedding(src) + self.src_position_embedding(src_positions)
        )
        encoder_out = self.encoder(x, src_mask)
        x = self.dropout(
            self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions)
        )
        decoder_out = self.decoder(x, encoder_out, src_mask, trg_mask)
        out = self.fc_out(decoder_out)
        return out