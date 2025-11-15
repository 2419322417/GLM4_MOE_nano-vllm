from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        # assert rotary_dim == head_size, (rotary_dim, head_size)
        self.rotary_dim = rotary_dim
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    # @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_rot, query_pass = query.split([self.rotary_dim, self.head_size - self.rotary_dim], dim=-1)
        key_rot, key_pass = key.split([self.rotary_dim, self.head_size - self.rotary_dim], dim=-1)

        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        
        query_rot = apply_rotary_emb(query_rot, cos, sin)
        key_rot = apply_rotary_emb(key_rot, cos, sin)
        query = torch.cat([query_rot, query_pass], dim=-1)
        key = torch.cat([key_rot, key_pass], dim=-1)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
    partial_rotary_factor: float = 1.0,
):
    assert rope_scaling is None
    if partial_rotary_factor < 1.0:
        rotary_dim = int(head_size * partial_rotary_factor)
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
