import torch
import torch.nn as nn
from safetensors import safe_open
from transformers.models.glm4_moe import Glm4MoeConfig
import torch.distributed as dist

from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.attention import Attention
from nanovllm.layers.RMSNorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, RowParallelLinear

class Glm4MoeAttention(nn.Module):
    def __init__(self, config: Glm4MoeConfig):
        super().__init__()
        tp_size = dist.get_world_size()
        self.config = config
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.total_num_heads

        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        
        # 定义 QKV 融合投影层
        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.attention_bias
        )
        
        # 定义输出投影层
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim, 
            self.hidden_size, 
            bias=False
        )
        # 根据配置决定是否使用 QK Norm
        self.use_qk_norm = getattr(config, "use_qk_norm", False)
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Rotary Embedding
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        rope_scaling = getattr(config, "rope_scaling", None)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=int(self.head_dim * partial_rotary_factor),
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            rope_scaling=rope_scaling,
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )


    # def load_weights(self, state_dict: dict[str, torch.Tensor], prefix: str):
    #     """从 state_dict 加载自己的权重,prefix 是权重在 state_dict 中的前缀。"""
    #     qkv_weight_name = f"{prefix}.qkv_proj.weight"
    #     if qkv_weight_name in state_dict:
    #         self.qkv_proj.weight.data.copy_(state_dict[qkv_weight_name])

    #     if self.config.attention_bias:
    #         qkv_bias_name = f"{prefix}.qkv_proj.bias"
    #         if qkv_bias_name in state_dict:
    #             self.qkv_proj.bias.data.copy_(state_dict[qkv_bias_name])

    #     o_proj_weight_name = f"{prefix}.o_proj.weight"
    #     if o_proj_weight_name in state_dict:
    #         self.o_proj.weight.data.copy_(state_dict[o_proj_weight_name])

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        if self.use_qk_norm:
            q = self.q_norm(q.view(-1, self.num_heads, self.head_dim))
            k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim))
        else:
            q = q.view(-1, self.num_heads, self.head_dim)
            k = k.view(-1, self.num_kv_heads, self.head_dim)
            
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output
        #return hidden_states