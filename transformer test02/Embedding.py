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