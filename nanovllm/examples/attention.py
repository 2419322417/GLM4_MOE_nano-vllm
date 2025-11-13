import torch

from transformers import AutoConfig, Glm4MoeConfig
from nanovllm.models.glm4_moe.attention_new import Glm4MoeAttention
from nanovllm.utils.context import set_context

def main():
    torch.manual_seed(42)
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.float16)

    # init_distributed_environment(world_size=1, rank=0)

    model = "/data/model/ZhipuAI/GLM-4.5-Air"
    # model = "/data/model/QuantTrio/GLM-4.6-AWQ"


    
    # model_index_file = f"{model}/model.safetensors.index.json" #question
    config = AutoConfig.from_pretrained(model)
    # quant_config = config.quantization_config
    #config = Glm4MoeConfig(config)
    # print(f"{config=}")
    # print(f"{quant_config=}")
    # print(config.num_kv_heads) 8
    # print(config.head_dim) 128
    prefix = "model.layers.0.self_attn"
    attn = Glm4MoeAttention(config, prefix=prefix)
    # attn = Glm4MoeAttention(config, prefix=prefix, quant_config=quant_config)

    load_weight=attn.load_weights(model, prefix)

    device = torch.device("cuda")
    #device = next(attn.parameters()).device
    import safetensors
    import os
    sample_path = "/data/ai_infra/debug/tensors1/rank_0"
    # sample_path = "/data/ai_infra/debug/glm4-6-awq-tensors"
    tensor_path = os.path.join(sample_path, f"{prefix}_2.safetensors")
    # tensor_path = os.path.join(sample_path, f"{prefix}_0.safetensors")
    loaded_tensor = safetensors.torch.load_file(tensor_path)
    hidden_states = loaded_tensor["hidden_states"].to(device=device)
    # print(f"{hidden_states.shape=}")
    # hidden_states = torch.randn(8192, 4096, device=device, dtype=torch.float16)
    # hidden_states = torch.randn(8192, 5120, device=device, dtype=torch.float16)
    kv_cache = {
        "k_cache": loaded_tensor["k_cache"].to(device=device),
        "v_cache": loaded_tensor["v_cache"].to(device=device)
    }
    attn.load_kv_cache(kv_cache)
    positions = loaded_tensor["positions"].to(device=device)
    output_reference = loaded_tensor["output"].to(device=device)
    # print(f"{hidden_states.shape=}, {positions.shape=}")
    slot_mapping=torch.arange(4, device='cuda:0', dtype=torch.int32)
    context_lens=None
    block_tables=None
    cu_seqlens_q=torch.tensor([ 0, 4], device='cuda:0', dtype=torch.int32)
    cu_seqlens_k=torch.tensor([ 0, 4], device='cuda:0', dtype=torch.int32)
    max_seqlen_q=4
    max_seqlen_k=4
    set_context(True,cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_k=max_seqlen_k, max_seqlen_q=max_seqlen_q, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
    output = attn(hidden_states, positions)
    # print(f"{output.shape=}")
    torch.testing.assert_close(output, output_reference, rtol=1e-2, atol=1e-2)
    # torch.testing.assert_close(output, output_reference, rtol=1e-3, atol=1e-3)
    print("完成对比！")

if __name__ == "__main__":
    main()
