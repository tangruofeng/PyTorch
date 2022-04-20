import torch.nn as nn
from Encoder import Encoder
from Decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size):
        super(Transformer, self).__init__()
        # tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6}
        self.encoder = Encoder(src_vocab_size)
        self.decoder = Decoder(tgt_vocab_size)
        self.projection = nn.Linear(512, tgt_vocab_size, bias=False)

    def forward(self, enc_inputs, dec_inputs):
        enc_outputs, enc_self_attention = self.encoder(enc_inputs)
        dec_outputs, dec_self_attention, dec_enc_attention = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs)
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attention, dec_self_attention, dec_enc_attention