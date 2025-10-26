import torch
import torch.nn as nn
from transformers.models.glm4_moe import Glm4MoeConfig

from .mlp import Glm4MoeMLP

class Glm4MoeMoE(nn.Module):
    def __init__(self, config: Glm4MoeConfig):
        super().__init__()
        self.config = config
        # self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        # self.experts = nn.ModuleList([Glm4MoeMLP(config) for _ in range(config.num_experts)])
        pass

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 暂时返回输入，以保证代码能运行
        return hidden_states