import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

class attention(nn.Module):
    def __init__(self):
        super(attention, self).__init__()

    def forward(self, query, key, value, mask):
        d_k = query.size(-1) #dk是矩阵的列数
        scores = torch.matmul(query, key.transpose(-1, -2)) / np.sqrt(d_k) # 得到QK‘
        #masked_fill方法有两个参数，maske和value，mask是一个pytorch张量（Tensor）
        #元素是布尔值，value是要填充的值
        #填充规则是mask中取值为True位置对应于self的相应位置用value填充
        scores = scores.masked_fill(mask==0,-1e9) # 使用mask矩阵遮挡单词
        attn = F.softmax(scores, dim=-1) # 计算 attention score
        return torch.matmul(attn, value), attn