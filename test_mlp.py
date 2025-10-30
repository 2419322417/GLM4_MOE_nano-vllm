import torch
import torch.nn as nn
import os
from typing import Optional
from nanovllm.models.glm4_moe.mlp import Glm4MoeMLP


def main():
    """
    在缺少vLLM导出数据时，生成随机样本测试Glm4MoeMLP输出形状是否正确
    """
    from transformers import AutoConfig

    torch.manual_seed(42)
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
    # 2. 初始化要测试的 MLP 层
    # -------------------------------------------------------------------------
    prefix = "model.layers.1.mlp"
    mlp = Glm4MoeMLP(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        hidden_act="silu",
        prefix=prefix,
    ).to("cuda").half()

    # -------------------------------------------------------------------------
    # 3. 构造随机输入样本
    # -------------------------------------------------------------------------
    batch_size = 2
    seq_len = 16
    x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=torch.float16)

    print(f"📦 随机输入张量 x 形状: {x.shape}")

    # -------------------------------------------------------------------------
    # 4. 前向推理
    # -------------------------------------------------------------------------
    print("🚀 开始执行前向推理...")
    output = mlp(x)
    print(f"✅ 输出张量 ret_x 形状: {output.shape}")

    # -------------------------------------------------------------------------
    # 5. 尺寸验证
    # -------------------------------------------------------------------------
    assert output.shape == x.shape, (
        f"❌ 输出尺寸不匹配! 输入: {x.shape}, 输出: {output.shape}"
    )
    print("🎯 输出尺寸与输入一致，MLP 层尺寸测试通过！")

#     # -------------------------------------------------------------------------
#     # （暂时注释掉误差比较部分）
#     # torch.testing.assert_close(output, output_reference, rtol=1e-3, atol=1e-3)
#     # print('✅ MLP 层输出与 vLLM 一致，验证通过！')


if __name__ == "__main__":
    main()
