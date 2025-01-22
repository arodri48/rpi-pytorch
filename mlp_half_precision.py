import torch
from torch import nn


class HalfLinear(nn.Linear):
    def __init__(self, in_features: int,
        out_features: int,
        bias: bool = True,
        dtype=torch.bfloat16):
        super().__init__(in_features, out_features, bias, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        converted_input = x.to(self.weight.dtype)
        result = super().forward(converted_input)
        return result.to(torch.float32)

class LinearBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float):
        super().__init__()
        self.fc = HalfLinear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class HalfMLP(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, out_features: int,
                 dropout: float = 0.5):
        super().__init__()
        self.model = nn.Sequential(
            LinearBlock(in_features, hidden_size, dropout),
            LinearBlock(hidden_size, hidden_size, dropout),
            LinearBlock(hidden_size, hidden_size, dropout),
            HalfLinear(hidden_size, out_features)
        )

    def forward(self, x):
        return self.model(x)