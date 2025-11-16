import os
from regex import P
import safetensors
import torch
from nanovllm.models.glm4_moe.moe import Glm4MoeMoE
from transformers import AutoConfig, Glm4MoeConfig



def main():
    torch.manual_seed(42) 
    torch.set_default_device("cuda")  
    torch.set_default_dtype(torch.float16)  

    model_path = "/data/model/ZhipuAI/GLM-4.5-Air"  
    config = AutoConfig.from_pretrained(model_path)  
    # print(f"{config=}")
    prefix = "model.layers.1.mlp"  

    moe = Glm4MoeMoE(config)

    moe.load_weights(model_path, prefix)  
    device = torch.device("cuda")
    #TODO 改成正确的路径
    sample_path = "/data/ai_infra/debug/tensors4/rank_0" 
    tensor_path = os.path.join(sample_path, f"{prefix}_7.safetensors")  
    loaded_tensor = safetensors.torch.load_file(tensor_path, device="cuda") 
    #TODO 改成正确的键值
    # print(loaded_tensor.keys())
    hidden_states = loaded_tensor["hidden_states_saved"].to(device=device)
    # hidden_states = torch.randn(8192, 4096, device="cuda", dtype=torch.float16)  
    # 加载参考输出用于验证
    output_reference = loaded_tensor["final_hidden_states"].to(device=device)

    output = moe(hidden_states)

    print(f"输入形状: {hidden_states.shape}")
    print(f"输出形状: {output.shape}")
    assert output.shape == hidden_states.shape, "输出维度与输入不匹配！"  
    torch.testing.assert_close(output, output_reference, rtol=1e-2, atol=1e-2),f"MoE输出与参考输出不匹配！"
    print("MoE测试通过！")

if __name__ == "__main__":
    main()

