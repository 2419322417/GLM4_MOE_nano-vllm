import torch.nn.functional as F
from sympy import residue
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

        # print(f"输入: {hidden_states.shape}")
        
        # hidden_states = hidden_states.view(-1, self.config.hidden_size)
        
        router_logits = F.linear(hidden_states, self.weight)
        scores = router_logits.sigmoid()
        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        
        topk_weights = scores.gather(1, topk_indices)
        
        denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
        topk_weights /= denominator
            
        topk_weights = topk_weights * self.routed_scaling_factor

        return topk_indices, topk_weights
    
class Glm4MoeMoE(nn.Module):
    def __init__(self,config:Glm4MoeConfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([
            Glm4MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act
                )
            for _ in range(config.n_routed_experts)
        ])
        self.gate = GlmMoeSelectTopk(config)
        self.shared_experts = Glm4MoeMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act
        )
        

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        topk_indices, topk_weights = self.gate(hidden_states)

        final_hidden_states = torch.zeros_like(hidden_states)

        for expert_idx, expert_layer in enumerate(self.experts):
            expert_mask = (topk_indices == expert_idx)
            if not expert_mask.any():
                continue

            token_indices = torch.where(expert_mask)[0]
            expert_weights = topk_weights[expert_mask]
            expert_input = hidden_states[token_indices]
            
            expert_output = expert_layer(expert_input)
            weighted_expert_output = expert_output * expert_weights.unsqueeze(-1)
            final_hidden_states.index_add_(0, token_indices, weighted_expert_output)

        hidden_states = final_hidden_states + self.shared_experts(hidden_states)
        hidden_states = hidden_states.view(batch_size, seq_len, hidden_size)
        return hidden_states


