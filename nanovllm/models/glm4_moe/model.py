import torch
import torch.nn as nn
from transformers.models.glm4_moe import Glm4MoeConfig
from safetensors.torch import load_file, safe_open
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

    def load_weights(self, model_path: str):
        """
        从模型路径加载 Glm4MoeModel 的权重。
        此方法会调用子模块的 load_weights 方法。
        """
        # 1. 加载词嵌入 (embedding)
        embed_key = "model.embed_tokens.weight"
        found = False
        safetensors_files = sorted(glob(os.path.join(model_path, "*.safetensors")))
        if not safetensors_files:
            raise FileNotFoundError(f"在目录 {model_path} 中未找到 .safetensors 文件")

        for f in safetensors_files:
            with safe_open(f, framework="pt", device="cpu") as sf:
                if embed_key in sf.keys():
                    self.embed_tokens.weight.data.copy_(sf.get_tensor(embed_key))
                    found = True
                    break
        if not found:
            raise ValueError(f"权重 '{embed_key}' 在路径 '{model_path}' 中未找到。")

        # 2. 循环调用每一层的加载方法
        for i, layer in enumerate(tqdm(self.layers, desc="加载解码层")):
            layer.load_weights(model_path, prefix=f"model.layers.{i}")

        # 3. 加载最后的 LayerNorm
        norm_key = "model.norm.weight"
        found = False
        for f in glob(os.path.join(model_path, "*.safetensors")):
            with safe_open(f, framework="pt", device="cpu") as sf:
                if norm_key in sf.keys():
                    self.norm.weight.data.copy_(sf.get_tensor(norm_key))
                    found = True
                    break
        if not found:
            raise ValueError(f"权重 '{norm_key}' 在路径 '{model_path}' 中未找到。")
    

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
        主加载函数：调用 self.model 的 load_weights，并加载 lm_head。
        """
        # 1. 调用模型主体的加载方法
        self.model.load_weights(model_path)

        # 2. 加载 LM Head (如果存在且不与词嵌入共享权重)
        if self.lm_head is not None and not self.config.tie_word_embeddings:
            lm_head_key = "lm_head.weight"
            found = False
            safetensors_files = sorted(glob(os.path.join(model_path, "*.safetensors")))
            if not safetensors_files:
                raise FileNotFoundError(f"在目录 {model_path} 中未找到 .safetensors 文件")

            for f in safetensors_files:
                with safe_open(f, framework="pt", device="cpu") as sf:
                    if lm_head_key in sf.keys():
                        self.lm_head.weight.data.copy_(sf.get_tensor(lm_head_key))
                        found = True
                        break
            if not found:
                # 注意：某些模型可能将 lm_head 存储在不同的文件中或使用不同的键名
                print(f"警告: 权重 '{lm_head_key}' 在路径 '{model_path}' 中未找到。")

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
