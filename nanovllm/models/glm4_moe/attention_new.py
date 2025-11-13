import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers.models.glm4_moe import Glm4MoeConfig
#import torch.distributed as dist

from nanovllm.distributed.parallel_state import get_tp_group
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.attention import Attention
from nanovllm.layers.RMSNorm import RMSNorm
# from nanovllm.layers.linear_awq import QKVParallelLinear, RowParallelLinear

class Glm4MoeAttention(nn.Module):
    def __init__(self, config: Glm4MoeConfig,prefix: str = "",quant_config: dict | None = None):
        super().__init__()
        tp_size = get_tp_group().world_size
        self.config = config
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.head_dim = config.head_dim

        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias=config.attention_bias

        self.quant_config=quant_config

        # 定义 QKV 融合投影层和输出投影层
        if quant_config is None:
            from nanovllm.layers.linear import QKVParallelLinear, RowParallelLinear
            self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=self.qkv_bias,
            )
            # print("Using non-quantized QKVParallelLinear")
        
            self.o_proj = RowParallelLinear(
                self.total_num_heads * self.head_dim, 
                self.hidden_size, 
                bias=False
            )
        else:
            from nanovllm.layers.linear_awq_new import QKVParallelLinear, RowParallelLinear
            
            self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=self.qkv_bias,
            quant_config=quant_config,
            )
            # print("Using AWQ quantized QKVParallelLinear")
        
            self.o_proj = RowParallelLinear(
                self.total_num_heads * self.head_dim, 
                self.hidden_size, 
                bias=False,
                quant_config=quant_config,  
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
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            rope_scaling=rope_scaling,
        )
        cache_config=None
        # quant_config=None
        # prefix=" "
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
        import os
        from safetensors import safe_open
        #print(f"Loading weights from directory {path} with prefix {prefix}")
        # print(self.total_num_heads)#96
        # print(self.head_dim)
        # print(self.hidden_size)#4096
        state_dict = {}
        if os.path.isdir(path):
            # 如果是目录，遍历所有 .safetensors 文件
            for filename in os.listdir(path):
                if filename.endswith(".safetensors"):
                    shard_path = os.path.join(path, filename)
                    # 使用 safe_open 安全地打开文件，先检查key，再加载tensor
                    with safe_open(shard_path, framework="pt", device="cpu") as f:
                        for key in f.keys():
                            if key.startswith(prefix):
                                # 只加载带有指定前缀的张量
                                state_dict[key] = f.get_tensor(key)
        else:
            # 如果是单个文件，直接加载
            #print(f"Loading weights from single file {path}")
            state_dict = load_file(path, device="cpu")

        if not state_dict:
            raise ValueError(f"No tensors with prefix '{prefix}' found in '{path}'")

        tp_group = get_tp_group()
        tp_rank = tp_group.rank
        tp_size = tp_group.world_size

        if self.quant_config is None:

            # 1. 加载并拆分 QKV 权重
            q_weight_key = f"{prefix}.q_proj.weight"
            k_weight_key = f"{prefix}.k_proj.weight"
            v_weight_key = f"{prefix}.v_proj.weight"

            if q_weight_key in state_dict and k_weight_key in state_dict and v_weight_key in state_dict:
                # 分别加载 Q, K, V 权重
                q_weight_global = state_dict[q_weight_key]
                k_weight_global = state_dict[k_weight_key]
                v_weight_global = state_dict[v_weight_key]

                # 按 TP rank 拆分 Q, K, V 各自的权重
                q_weight_tp = q_weight_global.split(self.q_size, dim=0)[tp_rank]
                k_weight_tp = k_weight_global.split(self.kv_size, dim=0)[tp_rank]
                v_weight_tp = v_weight_global.split(self.kv_size, dim=0)[tp_rank]
                # print(f"q_weight_tp.shape: {q_weight_tp.shape}")
                # print(f"k_weight_tp.shape: {k_weight_tp.shape}")
                # print(f"v_weight_tp.shape: {v_weight_tp.shape}")    
                # 拼接成当前 rank 所需的 QKV 权重
                qkv_weight_tp = torch.cat([q_weight_tp, k_weight_tp, v_weight_tp], dim=0)
                # print(qkv_weight_tp.shape)
                # print(self.qkv_proj.weight.shape)
                self.qkv_proj.weight.data.copy_(qkv_weight_tp)
            
            # 2. 加载并拆分 QKV 偏置 (如果存在)
            if self.config.attention_bias:
                q_bias_key = f"{prefix}.q_proj.bias"
                k_bias_key = f"{prefix}.k_proj.bias"
                v_bias_key = f"{prefix}.v_proj.bias"
                if q_bias_key in state_dict and k_bias_key in state_dict and v_bias_key in state_dict:
                    q_bias_global = state_dict[q_bias_key]
                    k_bias_global = state_dict[k_bias_key]
                    v_bias_global = state_dict[v_bias_key]

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
                # print(o_proj_weight_global.shape)
                chunk_size = o_proj_weight_global.shape[1] // tp_size
                o_proj_weight_tp = o_proj_weight_global.split(chunk_size, dim=1)[tp_rank]
                # print(f"o_proj_weight_tp.shape: {o_proj_weight_tp.shape}")
                # print(self.o_proj.weight.shape)
                self.o_proj.weight.data.copy_(o_proj_weight_tp)
        else:
            # 加载 AWQ 量化权重
            # 1. 加载并拆分 QKV 权重 (qweight, zeros, scales)
            bits = self.quant_config["bits"]
            pack_factor = 32 // bits 
            for tensor_name in ["qweight", "zeros", "scales"]:
                q_key = f"{prefix}.q_proj.{tensor_name}"
                k_key = f"{prefix}.k_proj.{tensor_name}"
                v_key = f"{prefix}.v_proj.{tensor_name}"

                if q_key in state_dict and k_key in state_dict and v_key in state_dict:
                    q_global = state_dict[q_key]
                    k_global = state_dict[k_key]
                    v_global = state_dict[v_key]

                    if tp_size > 1:
                        # AWQ qweight/zeros/scales 的形状是 [in_features, out_features_packed]
                        # QKVParallelLinear 将它们视为列并行，因此按 dim=1 拆分
                        q_tp = q_global.split(self.q_size // pack_factor, dim=1)[tp_rank]
                        k_tp = k_global.split(self.kv_size // pack_factor, dim=1)[tp_rank]
                        v_tp = v_global.split(self.kv_size // pack_factor, dim=1)[tp_rank]
                    else:
                        q_tp, k_tp, v_tp = q_global, k_global, v_global
  
                    qkv_tp = torch.cat([q_tp, k_tp, v_tp], dim=1)
                    # print(self.qkv_proj.qweight.shape)
                    # print(qkv_tp.shape)
                    getattr(self.qkv_proj, tensor_name).data.copy_(qkv_tp)

            # 2. 加载 QKV 偏置 
            if self.config.attention_bias:
                q_bias_key = f"{prefix}.q_proj.bias"
                k_bias_key = f"{prefix}.k_proj.bias"
                v_bias_key = f"{prefix}.v_proj.bias"
                if q_bias_key in state_dict and k_bias_key in state_dict and v_bias_key in state_dict:
                    q_bias_global = state_dict[q_bias_key]
                    k_bias_global = state_dict[k_bias_key]
                    v_bias_global = state_dict[v_bias_key]

                    if tp_size > 1:
                        q_bias_tp = q_bias_global.split(self.q_size, dim=0)[tp_rank]
                        k_bias_tp = k_bias_global.split(self.kv_size, dim=0)[tp_rank]
                        v_bias_tp = v_bias_global.split(self.kv_size, dim=0)[tp_rank]
                    else:
                        q_bias_tp = q_bias_global
                        k_bias_tp = k_bias_global
                        v_bias_tp = v_bias_global

                    qkv_bias_tp = torch.cat([q_bias_tp, k_bias_tp, v_bias_tp], dim=0)
                    self.qkv_proj.bias.data.copy_(qkv_bias_tp)

            # 3. 加载并拆分输出投影 (o_proj) 权重 (qweight, zeros, scales)
            for tensor_name in ["qweight", "zeros", "scales"]:
                o_proj_key = f"{prefix}.o_proj.{tensor_name}"
                if o_proj_key in state_dict:
                    o_proj_global = state_dict[o_proj_key]
                    if tp_size > 1:
                        # RowParallelLinear 按行拆分，即按 dim=0 拆分
                        chunk_size = o_proj_global.shape[0] // tp_size
                        o_proj_tp = o_proj_global.split(chunk_size, dim=0)[tp_rank]
                    else:
                        o_proj_tp = o_proj_global
                    # print(f"{tensor_name} shape: {o_proj_tp.shape}")
                    # print(f"{getattr(self.o_proj, tensor_name).shape=}")
                    getattr(self.o_proj, tensor_name).data.copy_(o_proj_tp)

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
        # print(hidden_states.shape)
        #测试代码
        # import safetensors
        # import os
        # sample_path = "/data/ai_infra/debug/tensors1/rank_0"
        # tensor_path = os.path.join(sample_path, f"model.layers.0.self_attn_2.safetensors")
        # loaded_tensor = safetensors.torch.load_file(tensor_path)
        # qkv_loaded = loaded_tensor["qkv"].to(device=hidden_states.device)
        # q_loaded = loaded_tensor["q"].to(device=hidden_states.device)
        # k_loaded = loaded_tensor["k"].to(device=hidden_states.device)

        #运行forward
        qkv = self.qkv_proj(hidden_states)
        # torch.testing.assert_close(qkv, qkv_loaded, rtol=1e-2, atol=1e-2)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # torch.testing.assert_close(q, q_loaded, rtol=1e-2, atol=1e-2)
        # torch.testing.assert_close(k, k_loaded, rtol=1e-2, atol=1e-2)
        # print(q.shape,k.shape,v.shape)
        if self.use_qk_norm:
            q = self.q_norm(q.view(-1, self.num_heads, self.head_dim))
            k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim))
        else:
            q = q.view(-1, self.num_heads, self.head_dim)
            k = k.view(-1, self.num_kv_heads, self.head_dim)
            
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        
        # print(q.shape,k.shape,v.shape)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output
        #return hidden_states
        
    def load_kv_cache(self, kv_cache: dict):
        self.attn.load_kv_cache(kv_cache)