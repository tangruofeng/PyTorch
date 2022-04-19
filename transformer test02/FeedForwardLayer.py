import torch.nn as nn

class FeedForwardLayer(nn.Module):
    def __init__(self):
        super(FeedForwardLayer, self).__init__()
        self.d_model = 512
        self.d_ff = 2048 # d_ff = 2048  # FeedForward dimension
        self.fc = nn.Sequential( # fc -> FeedForward Layer
            nn.Linear(self.d_model, self.d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(self.d_ff, self.d_model, bias=False),
        )

    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        # Add指 X+MultiHeadAttention(X)，是一种残差连接，通常用于解决多层网络训练的问题
        return nn.LayerNorm(self.d_model)(output + residual)
