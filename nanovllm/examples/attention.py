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
