import torch
import torch.nn as nn

from transformers.models.glm4_moe import Glm4MoeConfig

from nanovllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanovllm.layers.layernorm import RMSNorm
from .decode_layer import Glm4MoeDecoderLayer


class Glm4MoeModel(nn.Module):
    def __init__(self, config: Glm4MoeConfig):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        # self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.layers = nn.ModuleList([Glm4MoeDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        #self.layers = nn.ModuleList([])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        # print(f"11111 {hidden_states.shape=}")
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
            # pass
        # hidden_states, _ = self.norm(hidden_states, residual)
        hidden_states = self.norm(hidden_states)
        # print(f"22222 {hidden_states.shape=}")
        return hidden_states

class Glm4MoeForCausalLM(nn.Module):
    def __init__(self, config: Glm4MoeConfig):
        super().__init__()
        self.model = Glm4MoeModel(config)

        quant_config = None
        self.quant_config = quant_config

        def get_pp_group():
            class PPGroup:
                @property
                def is_last_rank(self):
                    return True
            return PPGroup()
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, quant_config=quant_config)
        else:
            self.lm_head = None

        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
