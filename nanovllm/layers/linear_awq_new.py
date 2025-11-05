import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import triton
import triton.language as tl

from nanovllm.distributed.parallel_state import get_tp_group

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_X': 64, 'BLOCK_SIZE_Y': 16}, num_warps=4, num_stages=1)
    ],
    key=['num_cols', 'num_rows', 'group_size'],
)
@triton.jit
def awq_dequantize_kernel(
        qweight_ptr,  # quantized matrix
        scales_ptr,  # scales, per group
        zeros_ptr,  # zeros, per group
        group_size,  # Should always be one of the supported group sizes
        result_ptr,  # Output matrix
        num_cols,  # input num cols in qweight
        num_rows,  # input num rows in qweight
        stride_wy, stride_wx,
        stride_sy, stride_sx,
        stride_zy, stride_zx,
        stride_oy, stride_ox,
        BLOCK_SIZE_X: tl.constexpr,
        BLOCK_SIZE_Y: tl.constexpr):
    # Setup the pids.
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)

    # Compute offsets and masks for qweight_ptr.
    offsets_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    offsets = offsets_y[:, None] * stride_wy + offsets_x[None, :] * stride_wx

    masks_y = offsets_y < num_rows
    masks_x = offsets_x < num_cols

    masks = masks_y[:, None] & masks_x[None, :]

    # Compute offsets and masks for result output ptr.
    result_offsets_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    result_offsets_x = pid_x * BLOCK_SIZE_X * 8 + tl.arange(
        0, BLOCK_SIZE_X * 8)
    result_offsets = (result_offsets_y[:, None] * stride_oy +
                      result_offsets_x[None, :] * stride_ox)

    result_masks_y = result_offsets_y < num_rows
    result_masks_x = result_offsets_x < num_cols * 8
    result_masks = result_masks_y[:, None] & result_masks_x[None, :]

    # Load the weights.
    iweights = tl.load(qweight_ptr + offsets, masks, 0.0)
    iweights = tl.interleave(iweights, iweights)
    iweights = tl.interleave(iweights, iweights)
    iweights = tl.interleave(iweights, iweights)
    # iweights = tl.reshape(iweights, (BLOCK_SIZE_Y * BLOCK_SIZE_X, 1))
    # iweights = tl.broadcast_to(iweights, (BLOCK_SIZE_Y * BLOCK_SIZE_X, 8))
    # iweights = tl.reshape(iweights, (BLOCK_SIZE_Y, BLOCK_SIZE_X * 8))

    # Create reverse AWQ order as tensor: [0, 4, 1, 5, 2, 6, 3, 7]
    # that will map given indices to the correct order.
    reverse_awq_order_tensor = tl.reshape((tl.arange(0, 2) * 4)[None, :] +
                                tl.arange(0, 4)[:, None], [8])

    # Use this to compute a set of shifts that can be used to unpack and
    # reorder the values in iweights and zeros.
    shifts = reverse_awq_order_tensor * 4
    shifts = tl.broadcast_to(shifts[None, :], (BLOCK_SIZE_Y * BLOCK_SIZE_X, 8))
    shifts = tl.reshape(shifts, (BLOCK_SIZE_Y, BLOCK_SIZE_X * 8))

    # Unpack and reorder: shift out the correct 4-bit value and mask.
    iweights = (iweights >> shifts) & 0xF

    # Compute zero offsets and masks.
    zero_offsets_y = pid_y * BLOCK_SIZE_Y // group_size + tl.arange(0, 1)
    zero_offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    zero_offsets = zero_offsets_y[:, None] * stride_zy + zero_offsets_x[None, :] * stride_zx

    zero_masks_y = zero_offsets_y < num_rows // group_size
    zero_masks_x = zero_offsets_x < num_cols
    zero_masks = zero_masks_y[:, None] & zero_masks_x[None, :]

    # Load the zeros.
    zeros = tl.load(zeros_ptr + zero_offsets, zero_masks, 0.0)
    zeros = tl.interleave(zeros, zeros)
    zeros = tl.interleave(zeros, zeros)
    zeros = tl.interleave(zeros, zeros)
    # zeros = tl.reshape(zeros, (BLOCK_SIZE_X, 1))
    # zeros = tl.broadcast_to(zeros, (BLOCK_SIZE_X, 8))
    # zeros = tl.reshape(zeros, (1, BLOCK_SIZE_X * 8))

    zeros = tl.broadcast_to(zeros, (BLOCK_SIZE_Y, BLOCK_SIZE_X * 8))

    # Unpack and reorder: shift out the correct 4-bit value and mask.
    zeros = (zeros >> shifts) & 0xF

    # Compute scale offsets and masks.
    scale_offsets_y = pid_y * BLOCK_SIZE_Y // group_size + tl.arange(0, 1)
    scale_offsets_x = (pid_x * BLOCK_SIZE_X * 8 +
                       tl.arange(0, BLOCK_SIZE_X * 8))
    scale_offsets = (scale_offsets_y[:, None] * stride_sy +
                     scale_offsets_x[None, :] * stride_sx)
    scale_masks_y = scale_offsets_y < num_rows // group_size
    scale_masks_x = scale_offsets_x < num_cols * 8
    scale_masks = scale_masks_y[:, None] & scale_masks_x[None, :]

    # Load the scales.
    scales = tl.load(scales_ptr + scale_offsets, scale_masks, 0.0)
    scales = tl.broadcast_to(scales, (BLOCK_SIZE_Y, BLOCK_SIZE_X * 8))

    # Dequantize.
    iweights = (iweights - zeros) * scales
    iweights = iweights.to(result_ptr.type.element_ty)

    # Finally, store.
    tl.store(result_ptr + result_offsets, iweights, result_masks)


# qweights - [K     , M // 8], int32
# scales   - [K // G, M     ], float16
# zeros    - [K // G, M // 8], int32
def awq_dequantize_triton(qweight: torch.Tensor,
                          scales: torch.Tensor,
                          zeros: torch.Tensor,
                          block_size_x: int = 32,
                          block_size_y: int = 32) -> torch.Tensor:
    K = qweight.shape[0]
    M = scales.shape[1]
    group_size = qweight.shape[0] // scales.shape[0]

    AWQ_TRITON_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]

    assert K > 0 and M > 0
    assert scales.shape[0] == K // group_size and scales.shape[1] == M
    assert zeros.shape[0] == K // group_size and zeros.shape[1] == M // 8
    assert group_size <= K
    assert group_size in AWQ_TRITON_SUPPORTED_GROUP_SIZES or group_size == K

    # Result tensor:
    # number of rows = same as input tensor
    # number of cols = 8 x input tensor num cols
    result = torch.empty(qweight.shape[0],
                         qweight.shape[1] * 8,
                         device=qweight.device,
                         dtype=scales.dtype)

    Y = qweight.shape[0]  # num rows
    X = qweight.shape[1]  # num cols

    grid = lambda META: (
        triton.cdiv(X, META['BLOCK_SIZE_X']),
        triton.cdiv(Y, META['BLOCK_SIZE_Y']),
    )
    # print(f'num_cols={X}, num_rows={Y}')
    # print(f'{qweight.stride(0)=}, {qweight.stride(1)=}, {scales.stride(0)=}, {scales.stride(1)=}, {zeros.stride(0)=}, {zeros.stride(1)=}, {result.stride(0)=}, {result.stride(1)=},')
    awq_dequantize_kernel[grid](
        qweight,
        scales,
        zeros,
        group_size,
        result,
        X,
        Y,
        qweight.stride(0), qweight.stride(1),
        scales.stride(0), scales.stride(1),
        zeros.stride(0), zeros.stride(1),
        result.stride(0), result.stride(1),
    )
    return result


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator




class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
        quant_config: dict | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = get_tp_group().rank
        self.tp_size = get_tp_group().world_size
        
        # self.group_size = group_size
        # self.bits = bits

        #self.weight = nn.Parameter(torch.empty(output_size, input_size))
        #self.weight.weight_loader = self.weight_loader
        self.quant_config=quant_config
        self.group_size = quant_config.get("group_size", 128)
        self.bits = quant_config.get("bits", 4)

        self.weight = nn.Parameter(torch.empty(
            input_size,
            output_size // (32 // self.bits),
            dtype=torch.int32
        ),requires_grad=False)

        self.scales = nn.Parameter(torch.empty(
            input_size // self.group_size,
            output_size
        ),requires_grad=False)

        # qzeros: [in_features // group_size, out_features // 8]
        self.qzeros = nn.Parameter(torch.empty(
            input_size // self.group_size,
            output_size // (32 // self.bits),
            dtype=torch.int32
        ),requires_grad=False)

        self.weight.weight_loader = self.weight_loader
        self.scales.weight_loader = self.weight_loader
        self.qzeros.weight_loader = self.weight_loader


        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_config: dict | None = None,
    ):
        super().__init__(input_size, output_size, bias, None, quant_config)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dequantized_weight = awq_dequantize_triton(self.weight, self.scales, self.qzeros)
        return F.linear(x, dequantized_weight.T, self.bias)


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_config: dict | None = None,
    ):
        tp_size = get_tp_group().world_size
        super().__init__(input_size, divide(output_size, tp_size), bias, 0, quant_config)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dequantized_weight = awq_dequantize_triton(self.weight, self.scales, self.qzeros)
        return F.linear(x, dequantized_weight.T, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
        quant_config: dict | None = None,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias, None, quant_config)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
        quant_config: dict | None = None,
    ):
        tp_size = get_tp_group().world_size
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias, quant_config)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_config: dict | None = None,
    ):
        tp_size = get_tp_group().world_size
        super().__init__(divide(input_size, tp_size), output_size, bias, 1, quant_config)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dequantized_weight = awq_dequantize_triton(self.weight, self.scales, self.qzeros)
        y = F.linear(x, dequantized_weight.T, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
