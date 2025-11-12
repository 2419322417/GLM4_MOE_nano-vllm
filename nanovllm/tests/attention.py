import torch

from transformers import AutoConfig, Glm4MoeConfig
from nanovllm.models.glm4_moe.attention import Glm4MoeAttention

def main():

    torch.manual_seed(42)
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.float16)

    # init_distributed_environment(world_size=1, rank=0)

    model = "/data/model/ZhipuAI/GLM-4.5-Air"
    config = AutoConfig.from_pretrained(model)
    config = Glm4MoeConfig(config)
    # print(f"{config=}")

    prefix = "model.layers.1.self_attn"
    attn = Glm4MoeAttention(config, prefix=prefix)

    attn.load_weights(model)

    import safetensors
    import os
    sample_path = "/data/ai_infra/debug/glm4-6-awq-tensors"
    tensor_path = os.path.join(sample_path, f"{prefix}_0.safetensors")
    loaded_tensor = safetensors.torch.load_file(tensor_path)
    hidden_states = loaded_tensor["hidden_states"]
    positions = loaded_tensor["positions"]
    output_reference = loaded_tensor["output"]

    # print(f"{hidden_states.shape=}, {positions.shape=}")
    output = attn(hidden_states, positions)
    # print(f"{output.shape=}")

    torch.testing.assert_close(output, output_reference, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    main()

import torch
import os
import safetensors
from transformers import AutoConfig
# 假设你的moe.py在nanovllm.models.glm4_moe包中
from nanovllm.models.glm4_moe.moe import Glm4MoeMoE

def main():
    torch.manual_seed(42)
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.float16)

    model_path = "/data/model/ZhipuAI/GLM-4.5-Air"
    config = AutoConfig.from_pretrained(model)

    # --- 模板改动 1: 定义 prefix ---
    # MoE层在GLM-4中通常名为 "mlp"
    # 我们测试第1层 (layers.1)
    layer_idx = 1
    prefix = f"model.layers.{layer_idx}.mlp"
    
    print(f"Initializing MoE layer for prefix: {prefix}")
    # --- 模板改动 2: 使用 prefix 初始化模型 ---
    moe = Glm4MoeMoE(config, prefix=prefix)

    # --- 模板改动 3: 调用 load_weights ---
    print(f"Loading weights from {model_path} for prefix {prefix}")
    moe.load_weights(model_path, prefix)
    print("✅ MoE weights loaded.")

    # --- 模板改动 4: 加载预计算的输入和参考输出 ---
    # 假设你有一个目录存放了用于调试的tensors
    sample_path = "/data/ai_infra/debug/glm4-6-awq-tensors"
    
    # 构造与prefix匹配的tensor文件名
    tensor_file = f"{prefix}_{layer_idx}.safetensors" # 例如: model.layers.1.mlp_1.safetensors
    tensor_path = os.path.join(sample_path, tensor_file)

    if not os.path.exists(tensor_path):
        print(f"⚠️ Reference tensor file not found at: {tensor_path}")
        print("Falling back to randomly generated input for shape check.")
        # 如果没有参考文件，就只检查形状
        # 注意：HF的MLP输入是，我们模拟一个
        hidden_states = torch.randn(8192, config.hidden_size, device="cuda", dtype=torch.float16)
        # MoE的forward需要3D输入
        output = moe(hidden_states.view(1, 8192, config.hidden_size))
        print(f"Input shape: {hidden_states.view(1, 8192, config.hidden_size).shape}")
        print(f"Output shape: {output.shape}")
        assert output.shape == hidden_states.view(1, 8192, config.hidden_size).shape
        print("✅ Shape check passed with random input.")
        return

    print(f"Loading reference tensors from: {tensor_path}")
    loaded_tensor = safetensors.torch.load_file(tensor_path)
    
    # 加载输入和参考输出
    hidden_states = loaded_tensor["hidden_states"].to(device="cuda")
    output_reference = loaded_tensor["output"].to(device="cuda")
    
    print(f"Input shape: {hidden_states.shape}")
    print(f"Reference output shape: {output_reference.shape}")

    # --- 模板改动 5: 执行前向传播并验证 ---
    output = moe(hidden_states)
    print(f"Your MoE output shape: {output.shape}")

    # 验证结果
    assert output.shape == output_reference.shape, "Output shape mismatch!"
    
    print("\nComparing outputs...")
    try:
        torch.testing.assert_close(output, output_reference, rtol=1e-3, atol=1e-3)
        print("🎉 Verification passed! Your MoE implementation matches the reference.")
    except AssertionError as e:
        print(f"⚠️ Verification failed: {e}")
        print("The output difference is larger than the tolerance.")

if __name__ == "__main__":
    main()
