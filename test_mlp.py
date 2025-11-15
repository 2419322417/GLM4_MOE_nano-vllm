import torch
import torch.nn as nn
import os
from safetensors.torch import load_file
from transformers import AutoConfig
from nanovllm.models.glm4_moe.mlp import Glm4MoeMLP
import torch.testing as tt

# def main():
#     """
#     使用从 FH_DEBUG 导出的张量文件测试 Glm4MoeMLP。
#     验证 forward 输出是否与保存时一致。
#     """

#     torch.set_default_device("cuda")
#     torch.set_default_dtype(torch.float16)

#     # -------------------------------------------------------------------------
#     # 1. 加载模型配置
#     # -------------------------------------------------------------------------
#     model = "/data/model/ZhipuAI/GLM-4.5-Air"
#     config = AutoConfig.from_pretrained(model)
#     hidden_size = config.hidden_size
#     intermediate_size = config.intermediate_size
#     print(f"✅ hidden_size = {hidden_size}, intermediate_size = {intermediate_size}")

#     # -------------------------------------------------------------------------
#     # 2. 指定 safetensors 数据文件路径
#     # -------------------------------------------------------------------------
#     tensor_path = "/data/ai_infra/debug/tensors1/rank_0/model.layers.0.mlp_2.safetensors"
#     assert os.path.exists(tensor_path), f"❌ 文件不存在: {tensor_path}"
#     print(f"📂 正在加载张量文件: {tensor_path}")

#     from safetensors import safe_open
#     with safe_open(tensor_path, framework="pt") as f:
#         x = f.get_tensor("x").to("cuda")
#         ref_output = f.get_tensor("ret_x").to("cuda")

#     print(f"📦 输入张量 hidden_states 形状: {x.shape}")
#     print(f"📦 参考输出张量 output 形状: {ref_output.shape}")

#     # -------------------------------------------------------------------------
#     # 3. 初始化要测试的 MLP 层
#     # -------------------------------------------------------------------------
#     # prefix = "model.layers.0.mlp"
#     prefix = "model.layers.0.mlp_1.safetensors"
#     mlp = Glm4MoeMLP(
#         hidden_size=hidden_size,
#         intermediate_size=intermediate_size,
#         hidden_act="silu",
#         prefix=prefix,
#     ).to("cuda").half()

#     # ✅ 加载对应 safetensors 权重
#     mlp.load_from_model(model, prefix)

#     # -------------------------------------------------------------------------
#     # 4. 前向推理
#     # -------------------------------------------------------------------------
#     print("🚀 开始执行前向推理...")
#     output = mlp(x)
#     print(f"✅ 推理输出张量形状: {output.shape}")

#     # -------------------------------------------------------------------------
#     # 5. 尺寸验证
#     # -------------------------------------------------------------------------
#     assert output.shape == ref_output.shape, (
#         f"❌ 输出尺寸不匹配! 模型输出: {output.shape}, 参考输出: {ref_output.shape}"
#     )

#     # -------------------------------------------------------------------------
#     # 6. 计算误差
#     # -------------------------------------------------------------------------
#     diff = torch.abs(output - ref_output)
#     max_diff = diff.max().item()
#     mean_diff = diff.mean().item()
#     print(f"📊 最大绝对误差: {max_diff:.6f}, 平均误差: {mean_diff:.6f}")

#     # 设置容忍阈值（FP16 精度）
#     if max_diff < 1e-2:
#         print("🎯 验证通过：输出与导出张量高度一致 ✅")
#     else:
#         print("⚠️ 注意：输出与保存的张量存在较大差异，请检查模型参数或量化配置。")


# if __name__ == "__main__":
#     main()








#---------------------逐层debug代码片段---------------------#
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
from safetensors.torch import load_file

from nanovllm.models.glm4_moe.mlp import Glm4MoeMLP

MODEL_DIR = "/data/model/ZhipuAI/GLM-4.5-Air"
TENSOR_PATH = "/data/ai_infra/debug/tensors3/rank_0/model.layers.0.mlp_4.safetensors"

HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 10944
PREFIX = "model.layers.0.mlp"

ERROR_THRESHOLD = 1e-2   # 你可以改成 1e-4 / 3e-3 等


def check_and_report(name, pred, ref, bad_layers):
    if pred.shape != ref.shape:
        print(f"❌ {name}: shape mismatch! pred={pred.shape}, ref={ref.shape}")
        bad_layers.append(name)
        return

    diff = (pred - ref).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    print(f"{name:>8}: max_err={max_err:.6f}, mean_err={mean_err:.6f}")

    if max_err > ERROR_THRESHOLD:
        bad_layers.append(name)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16  

    print(f"📂 加载张量文件: {TENSOR_PATH}")
    tensors = load_file(TENSOR_PATH)

    # 检查 keys
    needed = ["x", "gate_up", "gate", "up", "act_x", "down", "ret_x"]
    for k in needed:
        if k not in tensors:
            print(f"⚠️ 缺少张量: {k}")

    x_ref = tensors["x"].to(device=device, dtype=dtype)
    print(f"📦 输入 x 形状: {x_ref.shape}")

    # 加载 nano MLP
    mlp = Glm4MoeMLP(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        hidden_act="silu",
        quant_config=None,
        prefix=PREFIX,
    )
    mlp.load_from_model(MODEL_DIR, PREFIX)
    mlp.to(device=device, dtype=dtype)
    mlp.eval()

    bad_layers = []

    with torch.no_grad():
        print("gate_up_proj.weight shape:", mlp.gate_up_proj.weight.shape)
        W = mlp.gate_up_proj.weight.detach().cpu()

        # 切两个分支
        gate_W = W[:10944,:]
        up_W   = W[10944:,:]

        print("gate abs mean:", gate_W.abs().mean().item())
        print("up   abs mean:", up_W.abs().mean().item())






        # x
        # check_and_report("x", x_ref, x_ref, bad_layers)
        torch.testing.assert_close(x_ref, x_ref, rtol=1e-03, atol=1e-03)

        # gate_up
        gate_up_pred = mlp.gate_up_proj(x_ref)
        # check_and_report("gate_up", gate_up_pred, tensors["gate_up"].to(device, dtype), bad_layers)
        torch.testing.assert_close(gate_up_pred, tensors["gate_up"].to(device, dtype), rtol=1e-03, atol=1e-03)

        # gate / up
        gate_pred, up_pred = gate_up_pred.chunk(2, dim=-1)
        # print(f"gate_pred shape: {gate_pred.shape},gate_ref shape: {tensors['gate'].shape}")
        # print(f"up_pred shape: {up_pred.shape},up_ref shape: {tensors['up'].shape}")
        # check_and_report("gate", gate_pred, tensors["gate"].to(device, dtype), bad_layers)
        # check_and_report("up", up_pred, tensors["up"].to(device, dtype), bad_layers)
        torch.testing.assert_close(gate_pred, tensors["gate"].to(device, dtype), rtol=1e-03, atol=1e-03)
        torch.testing.assert_close(up_pred, tensors["up"].to(device, dtype), rtol=1e-03, atol=1e-03)

        # Silu(gate) * up
        gate_act_pred = torch.nn.functional.silu(gate_pred)
        act_x_pred = gate_act_pred * up_pred
        # check_and_report("act_x", act_x_pred, tensors["act_x"].to(device, dtype), bad_layers)
        torch.testing.assert_close(act_x_pred, tensors["act_x"].to(device, dtype), rtol=1e-03, atol=1e-03)

        # down_proj
        down_pred = mlp.down_proj(act_x_pred)
        # check_and_report("down", down_pred, tensors["down"].to(device, dtype), bad_layers)
        torch.testing.assert_close(down_pred, tensors["ret_x"].to(device, dtype), rtol=1e-03, atol=1e-03)

        # ret_x
        ret_x_pred = down_pred
        # check_and_report("ret_x", ret_x_pred, tensors["ret_x"].to(device, dtype), bad_layers)
        torch.testing.assert_close(ret_x_pred, tensors["ret_x"].to(device, dtype), rtol=1e-03, atol=1e-03)

    print("\n==============================")
    if len(bad_layers) == 0:
        print("🎉 所有层 **完全对齐**，无误差过大层！")
    else:
        print("❌ 以下层误差超过阈值:")
        for name in bad_layers:
            print(f"   - {name}")
        print("==============================")
        print("⚠️ 根据第一次出错的层，去检查对应算子 / 权重加载。")
    print("==============================\n")


if __name__ == "__main__":
    main()
# ------------------- Glm4MoeMLP 逐层debug代码片段 ------------------- #