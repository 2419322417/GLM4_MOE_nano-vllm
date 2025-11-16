import os
from safetensors import safe_open
import torch.nn.functional as F
import torch
import torch.nn as nn
# from transformers.models.glm4_moe import Glm4MoeConfig
# from transformers.models.glm4_moe import Glm4MoeModel
from .mlp import Glm4MoeMLP
from transformers.models.glm4_moe import Glm4MoeConfig
# from nanovllm.distributed.parallel_state import get_tp_group


class GlmMoeSelectTopk(nn.Module):
    def __init__(self, config: Glm4MoeConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        # self.n_group = config.n_group
        # self.topk_group = config.topk_group
        assert config.n_group == 1
        assert config.topk_group == 1
        # tp_size = get_tp_group()
        self.routed_scaling_factor = config.routed_scaling_factor
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size)))
        self.register_buffer("e_score_correction_bias", torch.zeros((self.n_routed_experts), dtype=torch.float32))
   
    def forward(self, hidden_states): 

        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        
        router_logits = F.linear(hidden_states, self.weight)
        # print("Router weight mean:", self.weight.mean().item())
        # print("Router weight std:", self.weight.std().item())
        # print("Router logits mean:", router_logits.mean().item())
        # print("Router logits std:", router_logits.std().item())
        scores = router_logits.sigmoid()
        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        
        topk_weights = scores.gather(1, topk_indices)
        
        denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
        topk_weights /= denominator
            
        topk_weights = topk_weights * self.routed_scaling_factor
        # print({topk_weights.shape},{topk_indices.shape}) #[4, 8]
        # print( topk_indices[:2]) #选前两个token选中的专家 [ 14,  55,  58,  61,  75,  90, 119,   2],

        return topk_indices, topk_weights
    
class Glm4MoeMoE(nn.Module):
    def __init__(self,config:Glm4MoeConfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([
            Glm4MoeMLP(
                hidden_size=config.hidden_size, # 4096
                intermediate_size=config.moe_intermediate_size, # 1408
                hidden_act=config.hidden_act
            )
            for _ in range(config.n_routed_experts)
        ])
        self.gate = GlmMoeSelectTopk(config)
        self.shared_experts = Glm4MoeMLP(
            hidden_size=config.hidden_size, # 4096
            intermediate_size=config.moe_intermediate_size, # 1408
            hidden_act=config.hidden_act
        )
        
    def load_weights(self, path: str, prefix: str):
        """
        加载MoE的权重：
        1. 路由模块（GlmMoeSelectTopk）的权重
        2. 所有专家（experts列表中的Glm4MoeMLP）的权重
        3. 共享专家（shared_experts的Glm4MoeMLP）的权重
        """
        # print(f"Loading weights from directory {path} with prefix {prefix}")
        state_dict = {}
        if os.path.isdir(path):
            for filename in os.listdir(path):
                if filename.endswith(".safetensors"):
                    with safe_open(os.path.join(path, filename), framework="pt", device="cpu") as f:
                        for key in f.keys():
                            if key.startswith(prefix):
                                state_dict[key] = f.get_tensor(key)
        else:
            # print(f"Loading weights from single file {path}")
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith(prefix):
                        state_dict[key] = f.get_tensor(key)

        if not state_dict:
            raise ValueError(f"No tensors with prefix '{prefix}' found in '{path}'")

        # 1. 加载路由模块的权重
        gate_weight_key = f"{prefix}.gate.weight"  #[128, 4096]
        if gate_weight_key in state_dict:
            self.gate.weight.data.copy_(state_dict[gate_weight_key].to(self.gate.weight.device))
            # print(f"路由权重 {gate_weight_key} 维度: {state_dict[gate_weight_key].shape}") 
        
        # 路由偏置量bias的：gate.e_score_correction_bias权重
        gate_bias_key = f"{prefix}.gate.e_score_correction_bias" #[128]
        if gate_bias_key in state_dict:   
            self.gate.e_score_correction_bias.copy_(state_dict[gate_bias_key].to(self.gate.e_score_correction_bias.device))
            # print(f"路由偏置 {gate_bias_key} 维度: {state_dict[gate_bias_key].shape}")

        # 2. 加载每个专家（experts）的权重
        for i, expert in enumerate(self.experts):
            expert_prefix = f"{prefix}.experts.{i}"
            # 加载专家内部的gate_up_proj和down_proj权重
            gate_proj_key = f"{expert_prefix}.gate_proj.weight"  #[1408, 4096]
            up_proj_key = f"{expert_prefix}.up_proj.weight"      #[1408, 4096]
            down_proj_key = f"{expert_prefix}.down_proj.weight"  #[4096, 1408]
            # print(f"专家 {i} gate_proj 维度: {state_dict[gate_proj_key].shape}")
            # print(f"专家 {i} up_proj 维度: {state_dict[up_proj_key].shape}")
            # print(f"专家 {i} down_proj 维度: {state_dict[down_proj_key].shape}")
            expert.load_weights(state_dict, expert_prefix)

        # 3. 加载共享专家（shared_experts）的权重
        shared_prefix = f"{prefix}.shared_experts"
        shared_gate_key = f"{shared_prefix}.gate_proj.weight"   #[1408, 4096]
        shared_up_key = f"{shared_prefix}.up_proj.weight"       #[1408, 4096]
        shared_down_key = f"{shared_prefix}.down_proj.weight"   #[1408, 4096]
        self.shared_experts.load_weights(state_dict, shared_prefix)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # print(f"输入hodden_states:{hidden_states.shape}")   #[1，4096]
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        topk_indices, topk_weights = self.gate(hidden_states)

        hidden_states_e= torch.zeros_like(hidden_states)

        for expert_idx, expert_layer in enumerate(self.experts):
            
            expert_mask = (topk_indices == expert_idx)
            # print(expert_mask.shape) #[4, 8]
            if not expert_mask.any():
                continue

            token_indices = torch.where(expert_mask)[0]
            weight_pos = torch.where(expert_mask)[1]  # 该专家在token的top-k中的位置
            expert_weights = topk_weights[token_indices, weight_pos]  #对应token的权重


            expert_input = hidden_states[token_indices]     
            expert_output = expert_layer(expert_input)
            weighted_expert_output = expert_output * expert_weights.unsqueeze(-1)

            hidden_states_e.index_add_(0, token_indices, weighted_expert_output)
        #TODO 验证共享专家，路由专家，最终输出是否正确
        # import safetensors
        # import os
        # sample_path = "/data/ai_infra/debug/tensors4/rank_0"
        # tensor_path = os.path.join(sample_path, f"model.layers.1.mlp_7.safetensors")
        # loaded_tensor = safetensors.torch.load_file(tensor_path)

        #测试专家输出是否正确
        # hidden_states_e_loaded = loaded_tensor["experts_output"].to(device=hidden_states_e.device)
        # torch.testing.assert_close(hidden_states_e, hidden_states_e_loaded, rtol=1e-2, atol=1e-2),f"专家输出不匹配"
        # print("MoE专家输出final_hidden_states形状:", hidden_states_e.shape)
        
        #测试共享专家输出是否正确
        b_shared_experts = self.shared_experts(hidden_states)
        # b_shared_experts_loaded = loaded_tensor["shared_output"].to(device=b_shared_experts.device)
        # torch.testing.assert_close(b_shared_experts, b_shared_experts_loaded, rtol=1e-2, atol=1e-2),f"共享专家输出不匹配"
        # print("MoE共享专家输出  final_hidden_states形状:", b_shared_experts.shape)

        # #测试最终输出是否正确
        hidden_states = hidden_states_e + b_shared_experts
        # hidden_states_loaded = loaded_tensor["final_hidden_states"].to(device=hidden_states.device)
        # torch.testing.assert_close(hidden_states, hidden_states_loaded, rtol=1e-2, atol=1e-2),f"最终输出不匹配"

        # print("输出hidden_states形状:", hidden_states.shape)
        return hidden_states        #[1, 4096]

