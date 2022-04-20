import torch.nn as nn
import torch
from MultiHeadAttention import MultiHeadAttention
from FeedForwardLayer import FeedForwardLayer
import Embedding

class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attention = MultiHeadAttention(512, 8)
        self.dec_enc_attention = MultiHeadAttention(512, 8)
        self.pos_ffn = FeedForwardLayer()

    # 包含两个 Multi-Head Attention 层。
    # 第一个 Multi-Head Attention 层采用了 Masked 操作。
    # 第二个 Multi-Head Attention 层的K, V矩阵使用 Encoder 的编码信息矩阵C进行计算，而Q使用上一个 Decoder block 的输出计算。
    # 最后有一个 Softmax 层计算下一个翻译单词的概率。
    def forward(self, dec_inputs, enc_outputs, dec_self_atte_mask, dec_env_atte_mask):
        dec_output, dec_self_attention = self.dec_self_attention(dec_inputs, dec_inputs, dec_inputs, dec_self_atte_mask)
        dec_output, dec_env_attention = self.dec_enc_attention(dec_output, enc_outputs, enc_outputs, dec_env_atte_mask) # K V 使用Encoder的outputs， Q使用Decoder第一个MHA的结果
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_self_attention, dec_env_attention

class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size):
        super(Decoder, self).__init__()
        # tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6}
        n_layers = 6
        self.target_embedding = nn.Embedding(tgt_vocab_size, 512)
        self.pos_embedding = nn.Embedding.from_pretrained(Embedding.positionEmbedding(6, 512),freeze=True)  # freeze参数 -> 固定住预训练模型，不再更新参数
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_output): # dec_inputs : [batch_size x target_len]
        # print(dec_inputs, enc_inputs, enc_output) # ]]], grad_fn=<NativeLayerNormBackward0>) tensor([[5, 1, 2, 3, 4]]) tensor([[1, 2, 3, 4, 0]])
        dec_outputs = self.target_embedding(dec_inputs) + self.pos_embedding(torch.LongTensor([[5, 1, 2, 3, 4]])) # Shifted right
        dec_self_attention_pad_mask = Embedding.get_attn_pad_mask(dec_inputs, dec_inputs)
        dec_self_attention_subsequent_mask = Embedding.get_attn_subsequent_mask(dec_inputs)
        # 逐个元素比较输入张量input是否大于另外的张量或浮点数other, 若大于则返回True，否则返回False，若张量other无法自动拓展成与输入张量input相同尺寸，则返回False
        dec_self_attention_mask = torch.gt((dec_self_attention_pad_mask + dec_self_attention_subsequent_mask), 0)
        dec_enc_attention_mask = Embedding.get_attn_pad_mask(dec_inputs, enc_inputs)
        dec_self_attention, dec_enc_attention = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_output, dec_self_attention_mask, dec_enc_attention_mask)
            dec_self_attention.append(dec_self_attn)
            dec_enc_attention.append(dec_enc_attn)
        return dec_outputs, dec_self_attention, dec_enc_attention