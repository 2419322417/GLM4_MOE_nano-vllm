import torch
import torch.nn as nn
#下面是从 transformers 库的 models.glm4_moe 模块中导入 Glm4MoeConfig 类，主要目的是加载或配置 GLM-4 MoE 模型的参数。
from transformers.models.glm4_moe import Glm4MoeConfig
from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.linear import MergedColumnParallelLinear, ColumnParallelLinear
from typing import Optional
from .mlp import Glm4MoeMLP #好像没用


class Glm4MoeMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        # nano版 替换QuantizationConfig对vllm的依赖。
        quant_config: Optional[object] = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()

        #nano 替换MergedColumnParallelLinear依赖
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,             # 输入维度
            2 * intermediate_size,   # 输出维度 = 两倍中间层
            bias=False
        )

        # nano版 替换RowParallelLinear依赖
        self.down_proj = ColumnParallelLinear(
            intermediate_size, 
            hidden_size, 
            bias=False
        )

        # 激活函数检查，为了代码健壮性，实则没啥用，就是为了防止传入奇怪的激活函数。
        if hidden_act.lower() != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )

    def forward(self, x):
        gate_up  = self.gate_up_proj(x) 
        act_x = self.act_fn(gate_up)
        ret_x = self.down_proj(act_x)

        return ret_x