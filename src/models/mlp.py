"""Преобразование признаков каждого токена внутри Transformer-блока."""

import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        if embed_dim <= 0 or hidden_dim <= 0:
            raise ValueError("embed_dim and hidden_dim must be positive")
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """[B,N,D] -> [B,N,D]"""
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
