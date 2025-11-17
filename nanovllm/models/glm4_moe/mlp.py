import torch
import torch.nn as nn
import os
from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.linear import MergedColumnParallelLinear, RowParallelLinear
from typing import Optional
from safetensors import safe_open
from safetensors.torch import load_file
from glob import glob


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
            hidden_size,
            [intermediate_size] * 2,  # ✅ 必须是列表，表示两个分支
            bias=False
        )

        # nano版 替换RowParallelLinear依赖
        self.down_proj = RowParallelLinear(
            intermediate_size, 
            hidden_size, 
            bias=False
        )

        # 激活函数检查，为了代码健壮性，实则没啥用，就是为了防止传入奇怪的激活函数。
        if hidden_act.lower() != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()
        self.prefix = prefix
        self.debug_id = 0

    def forward(self, x):
        gate_up  = self.gate_up_proj(x) 
        act_x = self.act_fn(gate_up)
        ret_x = self.down_proj(act_x)

        return ret_x

    @torch.no_grad()
    def forward_debug(self, x):
        """逐步返回所有中间结果，方便和 vLLM dump 的张量逐层对齐。"""
        # x: [B, hidden_size]
        out = {}
        out["x_in"] = x

        gate_up = self.gate_up_proj(x)           # [B, 2 * intermediate]
        out["gate_up"] = gate_up

        # MergedColumnParallelLinear: 两个分支拼在最后一维
        gate, up = gate_up.chunk(2, dim=-1)      # 各 [B, intermediate]
        out["gate"] = gate
        out["up"] = up

        # SiluAndMul: silu(gate) * up
        gate_act = torch.nn.functional.silu(gate)
        act_x = gate_act * up                    # [B, intermediate]
        out["act_x"] = act_x

        down = self.down_proj(act_x)             # [B, hidden_size]
        out["down"] = down
        out["ret_x"] = down                      # 和 forward 返回保持一致

        return out


    def load_weights(self, model_dir: str, prefix: str):
            """从 HF/GLM-4.5 权重中加载当前 MLP 层参数"""

            weight_files = sorted(glob(os.path.join(model_dir, "*.safetensors")))
            # print(f"📦 找到 {len(weight_files)} 个权重分片")

            gate_w = None
            up_w = None
            down_w = None

            for wf in weight_files:
                with safe_open(wf, framework="pt") as f:
                    for name in f.keys():
                        if not name.startswith(prefix):
                            continue

                        tensor = f.get_tensor(name)

                        if name.endswith("gate_proj.weight"):
                            # print(f"✅ 加载 {name} ({list(tensor.shape)})")
                            gate_w = tensor  # [intermediate, hidden]

                        elif name.endswith("up_proj.weight"):
                            # print(f"✅ 加载 {name} ({list(tensor.shape)})")
                            up_w = tensor    # [intermediate, hidden]

                        elif name.endswith("down_proj.weight"):
                            # print(f"✅ 加载 {name} ({list(tensor.shape)})")
                            down_w = tensor  # [hidden, intermediate]

            assert gate_w is not None, "gate_proj.weight 未找到"
            assert up_w is not None, "up_proj.weight 未找到"
            assert down_w is not None, "down_proj.weight 未找到"

            # 🔥 拼接 gate + up -> gate_up_proj
            gate_up = torch.cat([gate_w, up_w], dim=0)   # [2*intermediate, hidden]
            # print(f"📐 拼接后 gate_up 形状: {list(gate_up.shape)}")
            # print(f"📐 模块 gate_up_proj.weight 形状: {list(self.gate_up_proj.weight.shape)}")

            assert self.gate_up_proj.weight.shape == gate_up.shape, \
                f"gate_up_proj shape mismatch: module={self.gate_up_proj.weight.shape}, tensor={gate_up.shape}"

            # ✅ 不要转置，形状已经是 [out, in]
            self.gate_up_proj.weight.data.copy_(gate_up)

            # down_proj 也直接复制，不要转置
            assert self.down_proj.weight.shape == down_w.shape, \
                f"down_proj shape mismatch: module={self.down_proj.weight.shape}, tensor={down_w.shape}"
            self.down_proj.weight.data.copy_(down_w)

            # print("🎯 MLP 权重加载完成！")
    # def load_weights(self, state_dict: dict, prefix: str):
    #     """从state_dict中加载MLP的权重（gate_proj、up_proj、down_proj）"""
    #     gate_w = None
    #     up_w = None
    #     down_w = None

    #     for name, tensor in state_dict.items():
    #         if not name.startswith(prefix):
    #             continue

    #         if name.endswith("gate_proj.weight"):
    #             gate_w = tensor
    #         elif name.endswith("up_proj.weight"):
    #             up_w = tensor
    #         elif name.endswith("down_proj.weight"):
    #             down_w = tensor

    #     assert gate_w is not None, f"gate_proj.weight not found in prefix {prefix}"
    #     assert up_w is not None, f"up_proj.weight not found in prefix {prefix}"
    #     assert down_w is not None, f"down_proj.weight not found in prefix {prefix}"

    #     # 拼接gate和up的权重到gate_up_proj
    #     gate_up = torch.cat([gate_w, up_w], dim=0).to(self.gate_up_proj.weight.dtype)
    #     self.gate_up_proj.weight.data.copy_(gate_up)

    #     # 加载down_proj的权重
    #     down_w = down_w.to(self.down_proj.weight.dtype)
    #     self.down_proj.weight.data.copy_(down_w)
