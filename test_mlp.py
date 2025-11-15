import torch
import torch.nn as nn
import os
from safetensors.torch import load_file
from transformers import AutoConfig
from nanovllm.models.glm4_moe.mlp import Glm4MoeMLP


def main():
    """
    使用从 FH_DEBUG 导出的张量文件测试 Glm4MoeMLP。
    验证 forward 输出是否与保存时一致。
    """

    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.float16)

    # -------------------------------------------------------------------------
    # 1. 加载模型配置
    # -------------------------------------------------------------------------
    model = "/data/model/ZhipuAI/GLM-4.5-Air"
    config = AutoConfig.from_pretrained(model)
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    print(f"✅ hidden_size = {hidden_size}, intermediate_size = {intermediate_size}")

    # -------------------------------------------------------------------------
    # 2. 指定 safetensors 数据文件路径
    # -------------------------------------------------------------------------
    tensor_path = "/data/ai_infra/debug/tensors1/rank_0/model.layers.0.mlp_2.safetensors"
    assert os.path.exists(tensor_path), f"❌ 文件不存在: {tensor_path}"
    print(f"📂 正在加载张量文件: {tensor_path}")

    from safetensors import safe_open
    with safe_open(tensor_path, framework="pt") as f:
        x = f.get_tensor("x").to("cuda")
        ref_output = f.get_tensor("ret_x").to("cuda")

    print(f"📦 输入张量 hidden_states 形状: {x.shape}")
    print(f"📦 参考输出张量 output 形状: {ref_output.shape}")

    # -------------------------------------------------------------------------
    # 3. 初始化要测试的 MLP 层
    # -------------------------------------------------------------------------
    # prefix = "model.layers.0.mlp"
    prefix = "model.layers.0.mlp_1.safetensors"
    mlp = Glm4MoeMLP(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        hidden_act="silu",
        prefix=prefix,
    ).to("cuda").half()

    # ✅ 加载对应 safetensors 权重
    mlp.load_from_model(model, prefix)

    # -------------------------------------------------------------------------
    # 4. 前向推理
    # -------------------------------------------------------------------------
    print("🚀 开始执行前向推理...")
    output = mlp(x)
    print(f"✅ 推理输出张量形状: {output.shape}")

    # -------------------------------------------------------------------------
    # 5. 尺寸验证
    # -------------------------------------------------------------------------
    assert output.shape == ref_output.shape, (
        f"❌ 输出尺寸不匹配! 模型输出: {output.shape}, 参考输出: {ref_output.shape}"
    )

    # -------------------------------------------------------------------------
    # 6. 计算误差
    # -------------------------------------------------------------------------
    diff = torch.abs(output - ref_output)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"📊 最大绝对误差: {max_diff:.6f}, 平均误差: {mean_diff:.6f}")

    # 设置容忍阈值（FP16 精度）
    if max_diff < 1e-2:
        print("🎯 验证通过：输出与导出张量高度一致 ✅")
    else:
        print("⚠️ 注意：输出与保存的张量存在较大差异，请检查模型参数或量化配置。")


if __name__ == "__main__":
    main()
