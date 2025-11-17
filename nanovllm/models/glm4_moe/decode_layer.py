import torch
import torch.nn as nn
from transformers.models.glm4_moe import Glm4MoeConfig

from .attention_new import Glm4MoeAttention
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
        self.self_attn = Glm4MoeAttention(config, prefix = f"{prefix}.self_attn")
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
            self.mlp = Glm4MoeMoE(config)
        else:
            self.mlp = Glm4MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                prefix=f"{prefix}.mlp"
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def load_weights(self, model_path: str, prefix: str):
        """
        从模型路径加载权重。
        此方法会调用子模块的 load_weights 方法。
        """
        import os
        from safetensors.torch import load_file
        from safetensors import safe_open

        # 构造 layernorm 权重的完整名称
        input_ln_key = f"{prefix}.input_layernorm.weight"
        post_attn_ln_key = f"{prefix}.post_attention_layernorm.weight"

        # 遍历模型目录下的所有 safetensors 文件，查找并加载 layernorm 权重
        # 这种方式更高效，因为它只加载需要的张量，而不是整个文件
        loaded_keys = set()
        if os.path.isdir(model_path):
            for filename in sorted(os.listdir(model_path)):
                if filename.endswith(".safetensors"):
                    shard_path = os.path.join(model_path, filename)
                    with safe_open(shard_path, framework="pt", device="cpu") as f:
                        if input_ln_key in f.keys() and input_ln_key not in loaded_keys:
                            self.input_layernorm.weight.data.copy_(f.get_tensor(input_ln_key))
                            loaded_keys.add(input_ln_key)
                        if post_attn_ln_key in f.keys() and post_attn_ln_key not in loaded_keys:
                            self.post_attention_layernorm.weight.data.copy_(f.get_tensor(post_attn_ln_key))
                            loaded_keys.add(post_attn_ln_key)
        else: # 单个文件
            state_dict = load_file(model_path)
            if input_ln_key in state_dict:
                self.input_layernorm.weight.data.copy_(state_dict[input_ln_key])
            if post_attn_ln_key in state_dict:
                self.post_attention_layernorm.weight.data.copy_(state_dict[post_attn_ln_key])
        
        self.self_attn.load_weights(model_path, f"{prefix}.self_attn")
        self.mlp.load_weights(model_path, f"{prefix}.mlp")



    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # residual = hidden_states
        # hidden_states = self.input_layernorm(hidden_states)
        # hidden_states = self.self_attn(hidden_states, positions)
        # hidden_states = residual + hidden_states
        # residual = hidden_states
        # hidden_states = self.post_attention_layernorm(hidden_states)
        # hidden_states = self.moe(hidden_states)
        # hidden_states = residual + hidden_states
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)       
        hidden_states = self.self_attn(hidden_states, positions)        
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)        
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual