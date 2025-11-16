import torch
import os
from safetensors.torch import load_file
from nanovllm.models.glm4_moe.moe import Glm4MoeMLP


def main():
    """
    使用从 vLLM 导出的 safetensors 文件验证 Glm4MoeMLP 层一致性
    """
    from transformers import AutoConfig, Glm4MoeConfig

    torch.manual_seed(42)
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.float16)

    # -------------------------------------------------------------------------
    # 1. 加载模型配置
    # -------------------------------------------------------------------------
    model = "/data/model/ZhipuAI/GLM-4.5-Air"
    config = AutoConfig.from_pretrained(model)
    # config = Glm4MoeConfig(config)

    # -------------------------------------------------------------------------
    # 2. 初始化要测试的 MLP 层
    # -------------------------------------------------------------------------
    prefix = "model.layers.0.mlp" 
    # from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeMLP
    mlp = Glm4MoeMLP(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        hidden_act = config.hidden_act,
        prefix = prefix,
    )
    mlp.load_weights(model,prefix)
    print(f"gate_up_proj 权重尺寸: {mlp.gate_up_proj.weight.shape}")
    print(f"down_proj 权重尺寸: {mlp.down_proj.weight.shape}")

    # -------------------------------------------------------------------------
    # 3. 加载 vLLM 导出的参考输入与输出
    # -------------------------------------------------------------------------
    import safetensors
    sample_path = "/data/ai_infra/debug/tensors3/rank_0)"
    tensor_path = os.path.join(sample_path, f"{prefix}_0.safetensors")

    if not os.path.exists(tensor_path):
        raise FileNotFoundError(f"❌ 找不到文件: {tensor_path}")

    print(f"📦 正在加载调试张量: {tensor_path}")
    loaded_tensor = safetensors.torch.load_file(tensor_path)

    x = loaded_tensor["x"]  # MLP输入
    output_reference = loaded_tensor["ret_x"]  # vLLM输出（参考值）

    print(f"✅ 成功加载输入张量: {x.shape}")
    print(f"✅ 成功加载输出张量: {output_reference.shape}")

    # -------------------------------------------------------------------------
    # 4. 前向推理并比较结果
    # -------------------------------------------------------------------------
    print("🚀 开始执行前向推理...")
    output = mlp(x)

    print("🧮 比较输出结果...")
    torch.testing.assert_close(output, output_reference, rtol=1e-2, atol=1e-2)
    print("✅ MLP 层输出与 vLLM 一致，验证通过！")



if __name__ == "__main__":
    main()
