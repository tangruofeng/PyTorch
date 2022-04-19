import torch
import numpy as np

def positionEmbedding(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(max_len).unsqueeze(1)
    #div_term = torch.exp(pow(10000, torch.arange(0, d_model, 2) / d_model))
    #div_term = np.power(10000, 2 * (torch.arange(0, d_model, 2) // 2) / d_model)
    div_term = np.power(10000, (2 * torch.arange(0, d_model / 2, 1)) / d_model)
    pe[:, 0::2] = torch.sin(position / div_term)
    pe[:, 1::2] = torch.cos(position / div_term)
    pe = pe.unsqueeze(0)
    return pe

def get_attn_pad_mask(seq_q, seq_k):
    # 这个函数没看懂
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k

def get_attn_subsequent_mask(seq):
    attention_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # 当np.triu(a, k = 1)时，得到主对角线向上平移一个距离的对角线
    subsequent_mask = np.triu(np.ones(attention_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask
