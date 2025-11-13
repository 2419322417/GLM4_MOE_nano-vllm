import torch.nn as nn
from transformers.models.glm4_moe import Glm4MoeConfig

class Glm4MoeMLP(nn.Module):
    def __init__(self, config: Glm4MoeConfig, prefix: str = ""):
        super().__init__()
        # self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        # self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        # self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        # self.act_fn = nn.SiLU()
        pass
    def load_weights(self, model_path: str, prefix: str):
        pass

    def forward(self, hidden_states):
        # 暂时返回输入，以保证代码能运行
        return hidden_states