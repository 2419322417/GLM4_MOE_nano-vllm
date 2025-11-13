import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    # print(N)
    # print(slot_mapping.numel())
    assert key.stride(-1) == 1 and value.stride(-1) == 1 
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        cache_config=None,
        quant_config=None,
        prefix=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        # print(f"k_cache shape: {k_cache.shape}, v_cache shape: {v_cache.shape}")
        if k_cache.numel() and v_cache.numel():
            # print(k.shape)
            # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # print(context.slot_mapping)
            # print(context.context_lens)
            # print(context.block_tables)
            # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        # print(f"q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}")
        # print(f"k_cache shape: {k_cache.shape}, v_cache shape: {v_cache.shape}")
        # q shape: torch.Size([28, 16, 128]), k shape: torch.Size([28, 8, 128]), v shape: torch.Size([28, 8, 128])
        # k_cache shape: torch.Size([443, 256, 8, 128]), v_cache shape: torch.Size([443, 256, 8, 128])

        # block_tables = context.block_tables
        # gqa_k_cache = k_cache.unsqueeze(2).expand(-1, -1, 2, -1).reshape(*q.shape)
        # gqa_v_cache = v_cache.unsqueeze(2).expand(-1, -1, 2, -1).reshape(*q.shape)
        # score = torch.einsum("shd, shd -> hss")
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # print(context.slot_mapping)
            # print(context.context_lens)
            # print(context.block_tables)
            # print(context.max_seqlen_q)
            # print(context.cu_seqlens_q)
            # print(context.max_seqlen_k)
            # print(context.cu_seqlens_k)
            # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:
            # print(q.shape)
            # print(k_cache.shape)
            # print(v_cache.shape)
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables,
                                        softmax_scale=self.scale, causal=True)
        # o = torch.randn(*q.shape, device=q.device, dtype=q.dtype)
        return o
    def load_kv_cache(self, kv_cache: dict):
        self.k_cache = torch.zeros(443, 256, 8, 128)
        self.v_cache = torch.zeros(443, 256, 8, 128)
        # self.k_cache = kv_cache["k_cache"]
        # self.v_cache = kv_cache["v_cache"]