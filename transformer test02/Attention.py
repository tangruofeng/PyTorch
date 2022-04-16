import math
import torch
import torch.nn.functional as F

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1) #dk是矩阵的列数
    scores = torch.matmul(query, key.transpose(-1, 2)) / math.sqrt(d_k)

    if mask is not None:
        #masked_fill方法有两个参数，maske和value，mask是一个pytorch张量（Tensor）
        #元素是布尔值，value是要填充的值
        #填充规则是mask中取值为True位置对应于self的相应位置用value填充
        scores = scores.masked_fill(mask==0,-1e9)
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)
    return torch.matmul(scores, value), scores