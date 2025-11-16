import torch
from transformers import AutoConfig

from nanovllm.models.glm4_moe.decode_layer import Glm4MoeDecoderLayer


def main():
    """
    用于测试 Glm4MoeDecoderLayer 的示例函数。
    它会加载配置，实例化一个层，加载权重，并执行一次前向传播。
    """
    torch.manual_seed(42)
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.float16)

    # 请确保此路径指向你的 GLM-4 模型权重目录
    model_path = "/data/model/ZhipuAI/GLM-4.5-Air" 
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    # 指定要测试的层，例如第 0 层
    layer_idx = 0
    prefix = f"model.layers.{layer_idx}"
    
    print(f"Instantiating Glm4MoeDecoderLayer for layer {layer_idx}...")
    decoder_layer = Glm4MoeDecoderLayer(config, prefix=prefix)
    decoder_layer.to(torch.device("cuda"))

    print(f"Loading weights for layer {layer_idx} from '{model_path}'...")

    decoder_layer.load_weights(model_path, prefix)
    print("Weights loaded successfully.")

    # 创建伪造的输入张量以进行测试
    batch_size = 1
    num_tokens = 4
    hidden_states = torch.randn(
        batch_size * num_tokens, 
        config.hidden_size, 
        device="cuda", 
        dtype=torch.float16
    )

    positions = torch.arange(num_tokens, device="cuda").repeat(batch_size)
    residual = None
    print("Executing forward pass...")
    #decoder_layer.eval()
    with torch.no_grad():
        output, residual = decoder_layer(positions, hidden_states, residual)

    print("Forward pass completed.")
    print(f"Output shape: {output.shape}")

    # 检查输出形状是否与输入形状相同
    assert output.shape == hidden_states.shape, "Output shape does not match input shape!"
    print("Test passed: Output shape is correct.")


if __name__ == "__main__":
    main()