import torch
import torch.nn as nn
from MultiHeadAttention import MultiHeadAttention
from FeedForwardLayer import FeedForwardLayer
import Embedding

class EncoderLayer(nn.Module): # Add & Norm Layer
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attention = MultiHeadAttention(512, 8)
        self.pos_ffl = FeedForwardLayer()

    def forward(self, encoding_inputs, mask):
        encoding_outputs, self_attention = self.enc_self_attention(encoding_inputs, encoding_inputs, encoding_inputs, mask) # enc_inputs to same Q,K,V
        encoding_outputs = self.pos_ffl(encoding_outputs) # enc_outputs: [batch_size x len_q x d_model]
        return encoding_outputs, self_attention

class Encoder(nn.Module):
    def __init__(self, src_vocab_size):
        super(Encoder, self).__init__()
        n_layers = 6
        d_model = 512
        # src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4}
        # src_vocab_size = len(src_vocab)
        self.sr_embedding = nn.Embedding(src_vocab_size, d_model)
        self.pos_embedding = nn.Embedding.from_pretrained(Embedding.positionEmbedding(6, 512), freeze=True) # freeze参数 -> 固定住预训练模型，不再更新参数
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        enc_outputs = self.sr_embedding(enc_inputs) + self.pos_embedding(torch.LongTensor([[1, 2, 3, 4, 0]]))
        enc_self_attention_mask = Embedding.get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attention = []
        for layer in self.layers:
            enc_outputs, enc_self_att = layer(enc_outputs, enc_self_attention_mask)
            enc_self_attention.append(enc_self_att)
        # enc_outputs -> Encoder模块的结果
        # enc_self_attention -> 各层Encoder Layer中Attention 的Softmax层结果的拼接
        return enc_outputs, enc_self_attention