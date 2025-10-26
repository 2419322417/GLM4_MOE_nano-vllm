import torch
import torch.nn as nn
from transformers.models.glm4_moe import Glm4MoeConfig

from .attention import Glm4MoeAttention
from .moe import Glm4MoeMoE
from .mlp import Glm4MoeMLP
from nanovllm.layers.layernorm import RMSNorm

class Glm4MoeDecoderLayer(nn.Module):
    def __init__(
            self, 
            config: Glm4MoeConfig,
            prefix: str = "",
            ) -> None:
        super().__init__()
        self.self_attn = Glm4MoeAttention(config)
        layer_idx_str = prefix.split(sep=".")[-1]
        try:
            layer_idx = int(layer_idx_str)
        except ValueError:
            # 如果 prefix 为空或格式不正确，默认 layer_idx 为 0
            layer_idx = 0
        self.layer_idx = layer_idx
        if (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
        ):
            self.moe = Glm4MoeMLP(config)
        else:
            self.moe = Glm4MoeMoE(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        # residual = hidden_states
        # hidden_states = self.input_layernorm(hidden_states)
        # hidden_states = self.self_attn(hidden_states, positions)
        # hidden_states = residual + hidden_states
        # residual = hidden_states
        # hidden_states = self.post_attention_layernorm(hidden_states)
        # hidden_states = self.moe(hidden_states)
        # hidden_states = residual + hidden_states
        return hidden_states, residual