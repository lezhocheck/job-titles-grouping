from typing import Tuple
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super(ResidualBlock, self).__init__()
        self._fc = nn.Linear(hidden_dim, hidden_dim)
        self._bn = nn.BatchNorm1d(hidden_dim)
        self._relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self._relu(self._bn(self._fc(x)))


class SEBlock(nn.Module):
    def __init__(self, hidden_dim: int, reduction: int = 16) -> None:
        super(SEBlock, self).__init__()
        self._fc1 = nn.Linear(hidden_dim, hidden_dim // reduction)
        self._fc2 = nn.Linear(hidden_dim // reduction, hidden_dim)
        self._activation = nn.ReLU()
        self._sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=0, keepdim=True) 
        weights = self._sigmoid(self._fc2(self._activation(self._fc1(pooled))))
        return x * weights


class JobClassifier(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int,
        num_classes_level: int, 
        num_classes_area: int
    ) -> None:
        super(JobClassifier, self).__init__()
        self._shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            ResidualBlock(hidden_dim),  # Residual Block
            SEBlock(hidden_dim),  # Feature Attention
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3)
        )

        self._level_head = nn.Linear(hidden_dim, num_classes_level)
        self._area_head = nn.Linear(hidden_dim, num_classes_area) 
        
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shared_representation = self._shared_layers(x)
        level_output = self._level_head(shared_representation)
        area_output = self._area_head(shared_representation)
        return level_output, area_output