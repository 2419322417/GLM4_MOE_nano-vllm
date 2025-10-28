import torch
import torch.nn as nn
from transformers.models.glm4_moe import Glm4MoeConfig
from safetensors.torch import load_file
from tqdm import tqdm
import os
from glob import glob

from nanovllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanovllm.layers.layernorm import RMSNorm
from .decode_layer import Glm4MoeDecoderLayer


class Glm4MoeModel(nn.Module):
    def __init__(self, config: Glm4MoeConfig):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        # self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.layers = nn.ModuleList([Glm4MoeDecoderLayer(config,prefix=f"model.layers.{i}") for i in range(config.num_hidden_layers)])
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
    packed_modules_mapping = {
        "qkv_proj": ("qkv_proj", "qkv"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }
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
    def load_weights(self, model_path: str):
        """
        主加载函数：打开文件，分发加载任务。
        这是一个新添加的方法。
        """
        print(f"开始从目录 {model_path} 加载权重...")
        
        state_dict = {}
        safetensors_files = glob(os.path.join(model_path, "*.safetensors"))
        if not safetensors_files:
            raise FileNotFoundError(f"在目录 {model_path} 中未找到 .safetensors 文件")
            
        for file_path in tqdm(safetensors_files, desc="读取权重文件"):
            state_dict.update(load_file(file_path, device="cpu"))

        # 1. 加载词嵌入 (embedding)
        self.model.embed_tokens.weight.data.copy_(state_dict["model.embed_tokens.weight"])
        
        # 2. 循环调用每一层的加载方法
        for i, layer in enumerate(tqdm(self.model.layers, desc="加载解码层")):
            # prefix 已经在 __init__ 中传递，这里直接调用 load_weights
            layer.load_weights(state_dict, prefix=f"model.layers.{i}")
            
        # 3. 加载最后的 LayerNorm
        self.model.norm.weight.data.copy_(state_dict["model.norm.weight"])
        
        # 4. 加载 LM Head (如果存在)
        if self.lm_head is not None and not self.config.tie_word_embeddings:
             self.lm_head.weight.data.copy_(state_dict["lm_head.weight"])
        
        print("权重加载完成。")

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
