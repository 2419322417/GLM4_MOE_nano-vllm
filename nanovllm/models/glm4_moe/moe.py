from sys import prefix
import torch
import torch.nn as nn
from transformers.models.glm4_moe import Glm4MoeConfig

from .mlp import Glm4MoeMLP

def get_tensor_model_parallel_world_size():
    return 1  # 假设没有张量并行

def get_ep_group():
    class MockGroup:
        def rank(self): return 0
        def size(self): return 1
        def device_group(self): return self
    return MockGroup()

class Glm4MoeMoE(nn.Module):
    def __init__(self, config: Glm4MoeConfig):
        super().__init__()
        # self.tp_size = get_tensor_model_parallel_world_size()  #问题：为什么加载专家会爆，专家在哪里加载的
        # self.ep_group = get_ep_group().device_group
        self.ep_rank = self.ep_group.rank()
        self.ep_size = self.ep_group.size()
        self.routed_scaling_factor = config.routed_scaling_factor 
        self.n_routed_experts: int = config.n_routed_experts   #问题
        self.n_shared_experts: int = config.n_shared_experts

        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

        # 门控网络，给token分配专家
        self.gate = nn.Linear(
            config.hidden_size,
            config.n_routed_experts,
            bias=False
        )

        self.experts = nn.ModuleList([Glm4MoeMLP(config) for _ in range(config.n_routed_experts)])

        if self.n_shared_experts > 0:
            # 共享专家的中间层大小通常是单个专家的n_shared_experts倍 
            shared_intermediate_size = config.moe_intermediate_size * self.n_shared_experts
            self.shared_experts = Glm4MoeMLP(config, shared_intermediate_size)
        else:
            self.shared_experts = None



    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape()
        hidden_states = hidden_states.view(-1, hidden_dim)
        
        router_logits = self.gate(hidden_states.to(dtype=torch.float32))

        fused_moe_out = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )

        if self.shared_experts is not None:
            shared_output, final_hidden_states = fused_moe_out
            assert shared_output is not None
            final_hidden_states = (
                final_hidden_states * self.routed_scaling_factor + shared_output
            )
        else:
            final_hidden_states = fused_moe_out * self.routed_scaling_factor

        # 需要改laier
        # if self.tp_size > 1:
        #     final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(
        #         final_hidden_states
        #     )
        # return final_hidden_states.view(num_tokens, hidden_dim)


        return hidden_states.view(num_tokens, hidden_dim)