import torch
import math
import numpy as np

max_len=1000
#print(torch.arange(0, 1000, 2))
d_model = 512
pe = torch.zeros(max_len, d_model)
position = torch.arange(max_len).unsqueeze(1)
#div2_term = np.power(10000, 2 * (torch.arange(0, d_model, 2) // 2) / d_model)
div2_term = np.power(10000, (2 * torch.arange(0, d_model / 2, 1)) / d_model)
div_term = np.power(10000, 2 * ((torch.arange(0, d_model, 2) // 2)) / d_model)
#print(div2_term == div_term)
#print(torch.arange(0, d_model, 2) // 2)


mask = torch.arange(1,513,1)
n_heads = 512
#print(mask.shape)
mask = mask.unsqueeze(1).repeat(1, n_heads, 1, 1)# attn_mask : [batch_size x n_heads x len_q x len_k]
#print(mask)
#print(mask.shape)

seq_k = torch.tensor([0,1,0,2,3,4,5])
pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
t = pad_attn_mask.expand(512, 7, 7)
print(pad_attn_mask)
print(t[0])
print(t.shape)