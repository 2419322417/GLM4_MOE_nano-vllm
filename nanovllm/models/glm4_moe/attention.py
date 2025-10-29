import torch
import torch.nn as nn
from safetensors import load_file
from transformers.models.glm4_moe import Glm4MoeConfig
#import torch.distributed as dist

from nanovllm.distributed.parallel_state import get_tp_group
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.attention import Attention
from nanovllm.layers.RMSNorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, RowParallelLinear

class Glm4MoeAttention(nn.Module):
    def __init__(self, config: Glm4MoeConfig):
        super().__init__()
        tp_size = get_tp_group().world_size
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
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 0.5)
        rope_scaling = getattr(config, "rope_scaling", None)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=int(self.head_dim * partial_rotary_factor),
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            rope_scaling=rope_scaling,
        )
        cache_config=None
        quant_config=None
        prefix=" "
#todo attention
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )


    def load_weights(self, path: str, prefix: str):
        """
        从 Safetensors 文件加载权重，并根据张量并行（TP）进行拆分。
        Args:
            path (str): Safetensors 权重文件的路径。
            prefix (str): 当前注意力层在权重文件中的前缀 (例如 "transformer.layers.0.self_attn")。
        """
        state_dict = load_file(path, device="cpu")
        tp_group = get_tp_group()
        tp_rank = tp_group.rank
        tp_size = tp_group.world_size

        # 1. 加载并拆分 QKV 权重
        qkv_weight_key = f"{prefix}.qkv_proj.weight"
        if qkv_weight_key in state_dict:
            # 获取完整的 QKV 权重
            qkv_weight = state_dict[qkv_weight_key]
            
            # 计算 Q, K, V 各自的总维度
            total_q_dim = self.total_num_heads * self.head_dim
            total_kv_dim = self.total_num_kv_heads * self.head_dim

            # 从完整权重中分离出 Q, K, V
            q_weight_global = qkv_weight[:total_q_dim]
            k_weight_global = qkv_weight[total_q_dim : total_q_dim + total_kv_dim]
            v_weight_global = qkv_weight[total_q_dim + total_kv_dim:]

            # 按 TP rank 拆分 Q, K, V 各自的权重
            q_weight_tp = q_weight_global.split(self.q_size, dim=0)[tp_rank]
            k_weight_tp = k_weight_global.split(self.kv_size, dim=0)[tp_rank]
            v_weight_tp = v_weight_global.split(self.kv_size, dim=0)[tp_rank]

            # 拼接成当前 rank 所需的 QKV 权重
            qkv_weight_tp = torch.cat([q_weight_tp, k_weight_tp, v_weight_tp], dim=0)
            self.qkv_proj.weight.data.copy_(qkv_weight_tp)

        # 2. 加载并拆分 QKV 偏置 (如果存在)
        if self.config.attention_bias:
            qkv_bias_key = f"{prefix}.qkv_proj.bias"
            if qkv_bias_key in state_dict:
                qkv_bias = state_dict[qkv_bias_key]
                total_q_dim = self.total_num_heads * self.head_dim
                total_kv_dim = self.total_num_kv_heads * self.head_dim

                q_bias_global = qkv_bias[:total_q_dim]
                k_bias_global = qkv_bias[total_q_dim : total_q_dim + total_kv_dim]
                v_bias_global = qkv_bias[total_q_dim + total_kv_dim:]

                q_bias_tp = q_bias_global.split(self.q_size, dim=0)[tp_rank]
                k_bias_tp = k_bias_global.split(self.kv_size, dim=0)[tp_rank]
                v_bias_tp = v_bias_global.split(self.kv_size, dim=0)[tp_rank]

                qkv_bias_tp = torch.cat([q_bias_tp, k_bias_tp, v_bias_tp], dim=0)
                self.qkv_proj.bias.data.copy_(qkv_bias_tp)

        # 3. 加载并拆分输出投影 (o_proj) 权重
        o_proj_weight_key = f"{prefix}.o_proj.weight"
        if o_proj_weight_key in state_dict:
            o_proj_weight_global = state_dict[o_proj_weight_key]
            # RowParallelLinear 按列拆分，即按 dim=1 拆分
            chunk_size = o_proj_weight_global.shape[1] // tp_size
            o_proj_weight_tp = o_proj_weight_global.split(chunk_size, dim=1)[tp_rank]
            self.o_proj.weight.data.copy_(o_proj_weight_tp)

        # 4. 加载 QK Norm 权重 (如果存在)
        if self.use_qk_norm:
            q_norm_key = f"{prefix}.q_norm.weight"
            if q_norm_key in state_dict:
                self.q_norm.weight.data.copy_(state_dict[q_norm_key])
            
            k_norm_key = f"{prefix}.k_norm.weight"
            if k_norm_key in state_dict:
                self.k_norm.weight.data.copy_(state_dict[k_norm_key])

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