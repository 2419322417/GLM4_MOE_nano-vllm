import torch
import torch.nn.functional as F
from nanovllm.layers.linear_awq_new import ColumnParallelLinear

def reference_dequantize(qweight, scales, qzeros, group_size, bits=4):
    """
    纯 PyTorch 实现的参考反量化函数，用于验证。
    """
    assert bits == 4, "此参考实现仅支持 4-bit"
    
    # 1. 准备参数和形状
    input_size, _ = qweight.shape
    output_size = scales.shape[1]
    
    order = [i + j * 4 for i in range(4) for j in range(2)]

    # 2. 解包零点 (Unpack Zeros)
    # qzeros 是 int32, 每 32 位包含 8 个 4-bit 零点
    unpacked_zeros = torch.zeros(
        (input_size // group_size, output_size), 
        dtype=torch.int32, 
        device=qzeros.device
    )
    for i in range(8):
        # 使用正确的顺序 order[i] 来提取
        unpacked_zeros[:, i::8] = (qzeros >> (order[i] * 4)) & 0xF
  
    
    # 扩展零点以匹配权重的形状
    zeros = unpacked_zeros.repeat_interleave(group_size, dim=0)
    
    # 3. 解包权重 (Unpack Weights)
    # qweight 是 int32, 每 32 位包含 8 个 4-bit 权重
    unpacked_weights = torch.zeros(
        (input_size , output_size), 
        dtype=torch.int32, 
        device=qweight.device
    )
    for i in range(8):
        unpacked_weights[:, i::8] = (qweight >> (order[i] * 4)) & 0xF
        
    # 堆叠并重塑为 (input_size, output_size)
 

    # 4. 扩展缩放 (Expand Scales)
    # scales 的形状是 (input_size / group_size, output_size)
    # 需要扩展以匹配权重的形状
    scales = scales.repeat_interleave(group_size, dim=0)

    # 5. 应用反量化公式: (weight - zero) * scale
    dequantized_weight = (unpacked_weights.to(scales.dtype) - zeros.to(scales.dtype)) * scales
    
    
    return dequantized_weight

def test():
    """
    执行 ColumnParallelLinear 的反量化验证测试。
    """
    #设置测试参数
    input_size = 1024
    output_size = 1024
    group_size = 128
    bits = 4
    
    batch_size = 2
    seq_len = 8
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    quant_config={
            "group_size": 128,
            "bits": 4
        }

    print(f"在设备上运行测试: {device}")

    # --- 2. 创建模拟数据 ---
    # 输入张量
    x = torch.randn(batch_size * seq_len, input_size, dtype=torch.float16, device=device)

    # 量化权重 (qweight)
    # 形状: (input_size, output_size // (32 // bits))
    # qweight = torch.randint(0, 2**32, (input_size, output_size // (32 // bits)), dtype=torch.int64, device=device).to(torch.int32)

    qweight = torch.randint(-2**31, 2**31-1, (input_size, output_size // (32 // bits)), dtype=torch.int32, device=device)

    # 缩放 (scales)
    # 形状: (input_size // group_size, output_size)
    scales = torch.randn((input_size // group_size, output_size), dtype=torch.float16, device=device)

    # 量化零点 (qzeros)
    # 形状: (input_size // group_size, output_size // (32 // bits))
    # 每个 int32 包含 8 个 4-bit 零点
    # qzeros = torch.randint(0, 2**32 , (input_size // group_size, output_size // (32 // bits)), dtype=torch.int64, device=device).to(torch.int32)
    qzeros = torch.randint(-2**31, 2**31-1, (input_size // group_size, output_size // (32 // bits)), dtype=torch.int32, device=device)

    # --- 3. 使用 ColumnParallelLinear 进行计算 ---
    # 实例化模块
    # 注意: 由于 world_size=1, output_size 不需要改变
    awq_linear_layer = ColumnParallelLinear(
        input_size=input_size,
        output_size=output_size,
        bias=False,
        quant_config=quant_config
    ).to(device).half()

    # 手动加载权重
    awq_linear_layer.weight.data.copy_(qweight)
    awq_linear_layer.scales.data.copy_(scales)
    awq_linear_layer.qzeros.data.copy_(qzeros)

    # 执行前向传播
    # 这将调用内部的 awq_dequantize_triton 内核
    output_triton = awq_linear_layer(x)

    # --- 4. 使用参考方法进行计算 ---
    # 使用参考函数进行反量化
    ref_dequantized_weight = reference_dequantize(qweight, scales, qzeros, group_size, bits)
    
    # 使用标准 F.linear 计算
    # 注意: 我们的权重是 (in, out) 排列，F.linear 需要 (out, in)
    output_reference = F.linear(x, ref_dequantized_weight.T)

    # --- 5. 验证结果 ---
    print("Triton 内核输出的形状:", output_triton.shape)
    print("参考实现输出的形状:", output_reference.shape)

    # 比较两个输出张量
    are_close = torch.allclose(output_triton, output_reference, atol=1e-2, rtol=1e-2)

    if are_close:
        print("\n 测试通过: 结果一致。")
    else:
        print("\n 测试失败: 结果不一致。")
        diff = torch.abs(output_triton - output_reference)
        print(f"  - 最大绝对误差: {diff.max().item()}")
        print(f"  - 平均绝对误差: {diff.mean().item()}")

if __name__ == "__main__":
    test()