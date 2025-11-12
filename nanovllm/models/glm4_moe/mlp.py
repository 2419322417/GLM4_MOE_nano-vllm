import torch
import torch.nn as nn
import os
from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.linear import MergedColumnParallelLinear, RowParallelLinear
from typing import Optional
from safetensors import safe_open
from safetensors.torch import load_file



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
    

    def load_from_model(self, model_dir: str, prefix: str):
        """从 vLLM 导出的 safetensors 权重中加载当前 MLP 层参数"""
        import glob
        weight_files = sorted(glob.glob(f"{model_dir}/*.safetensors"))
        print(f"📦 找到 {len(weight_files)} 个权重分片")

        # 遍历所有权重文件，找到对应层的参数
        for wf in weight_files:
            tensors = load_file(wf)
            for name, tensor in tensors.items():
                if name.startswith(prefix):
                    if "gate_up_proj" in name:
                        print(f"✅ 加载 {name}")
                        self.gate_up_proj.weight.data.copy_(tensor)
                    elif "down_proj" in name:
                        print(f"✅ 加载 {name}")
                        self.down_proj.weight.data.copy_(tensor)
    
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
    config = Glm4MoeConfig(config)

    # -------------------------------------------------------------------------
    # 2. 初始化要测试的 MLP 层
    # -------------------------------------------------------------------------
    prefix = "model.layers.1.mlp"  # 选择对应层（可以更改为其他层进行验证）
    from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeMLP
    mlp = Glm4MoeMLP(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        hidden_act = config.hidden_act,
        prefix = prefix,
    )
    mlp.load_weights(model)

    # -------------------------------------------------------------------------
    # 3. 加载 vLLM 导出的参考输入与输出
    # -------------------------------------------------------------------------
    import safetensors
    sample_path = "/data/ai_infra/debug/glm4-6-awq-tensors"
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
    torch.testing.assert_close(output, output_reference, rtol=1e-3, atol=1e-3)
    print("✅ MLP 层输出与 vLLM 一致，验证通过！")

if __name__ == "__main__":
    main()