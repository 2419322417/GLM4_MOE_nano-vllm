from sys import prefix
import torch
import torch.nn as nn
from transformers.models.glm4_moe import Glm4MoeConfig

from vllm.model_executor.layers.fused_moe import fused_moe

from .mlp import Glm4MoeMLP

def get_tensor_model_parallel_world_size():
    return 1  # 假设没有张量并行

def get_ep_group():
    class MockGroup:
        def rank(self): return 0
        def size(self): return 1
        def device_group(self): return self
    return MockGroup()

class ExpertRunner(nn.Module):
    def __init__(self, experts: nn.ModuleList):
        super().__init__()
        self.experts = experts
    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        if not hasattr(self, 'w1'):
            print("Preparing and caching MoE weights...")
            w1_list = [expert.dense_h_to_4h.weight for expert in self.experts]
            w2_list = [expert.dense_4h_to_h.weight for expert in self.experts]
            w3_list = [expert.dense_h_to_4h_2.weight for expert in self.experts]
            self.register_buffer('w1', torch.stack(w1_list, dim=0))
            self.register_buffer('w2', torch.stack(w2_list, dim=0))
            self.register_buffer('w3', torch.stack(w3_list, dim=0))
            print("MoE weights cached.")

        return fused_moe(
            hidden_states=hidden_states,
            w1=self.w1, w2=self.w2, w3=self.w3,
            topk=2, renormalize=True, gating_output=router_logits
        )

class Glm4MoE(nn.Module):
    def __init__(self, config: Glm4MoeConfig):
        super().__init__()
        ep_group = get_ep_group()
        self.ep_rank = ep_group.rank()
        self.ep_size = ep_group.size()
        # self.tp_size = get_tensor_model_parallel_world_size()  
        # self.ep_group = get_ep_group().device_group
  
        self.routed_scaling_factor = config.routed_scaling_factor 
        self.n_routed_experts: int = config.n_routed_experts   
        self.n_shared_experts: int = config.n_shared_experts

        # 门控网络，给token分配专家
        self.gate = nn.Linear(
            config.hidden_size,
            config.n_routed_experts,
            bias=False
        )

        expert_list = [Glm4MoeMLP(config) for _ in range(config.n_routed_experts)]
        self.experts = ExpertRunner(nn.ModuleList(expert_list))

        if self.n_shared_experts > 0:
            shared_intermediate_size = config.moe_intermediate_size * self.n_shared_experts
            self.shared_experts = Glm4MoeMLP(config, intermediate_size=shared_intermediate_size)
        else:
            self.shared_experts = None

    def load_weights(self, model_path: str, prefix: str):
        pass

    def load_weights(self, model_path: str, prefix: str):
        pass

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
         router_logits = self.gate(hidden_states)

         routed_expert_output = self.experts(hidden_states=hidden_states, router_logits=router_logits)
        
         final_hidden_states = routed_expert_output * self.routed_scaling_factor 

         if self.shared_experts is not None:
            shared_output = self.shared_experts(hidden_states)
            final_hidden_states = final_hidden_states + shared_output

        # 需要改laier
        # if self.tp_size > 1:
        #     final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(
        #         final_hidden_states
        #     )
        # return final_hidden_states.view(num_tokens, hidden_dim)


         return final_hidden_states