import torch
import torch.nn as nn
from transformers.models.glm4_moe import Glm4MoeConfig

class Glm4MoeAttention(nn.Module):
    def __init__(self, config: Glm4MoeConfig):
        super().__init__()
        self.config = config
        # self.qkv_proj = nn.Linear(...)
        # self.o_proj = nn.Linear(...)
        pass

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return hidden_states