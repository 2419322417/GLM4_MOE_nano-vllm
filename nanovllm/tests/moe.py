from regex import P
import torch
from nanovllm.models.glm4_moe.moe import Glm4MoeMoE
import torch
from transformers import AutoConfig, Glm4MoeConfig

def main():

    torch.manual_seed(42)
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.float16)
   
    model = "/data/model/ZhipuAI/GLM-4.5-Air"  
    config = AutoConfig.from_pretrained(model)
    # config = Glm4MoeConfig(config)
    # print(f"{config=}")

    prefix = "model.layers.1.self_mlp"
    moe = Glm4MoeMoE(config)
    batch_size = 1 
    seq_len = 4
    hidden_size = config.hidden_size
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    print(f"输入:{hidden_states.shape}") 
    output = moe(hidden_states)
    print(f"输出：{output.shape}")

if __name__ == "__main__":
    main()
