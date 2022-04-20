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
    return torch.FloatTensor(pe)

def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)

p = 6
d = 512
print(positionEmbedding(p, d))
print(get_sinusoid_encoding_table(p, d))
for each in positionEmbedding(p, d) == get_sinusoid_encoding_table(p, d):
    print(each)