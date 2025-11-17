import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    # @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        #todo: 温度采样,temperature == 0时为贪心采样
        if torch.all(temperatures == 0):
            # 贪心采样
            sample_tokens = torch.argmax(logits, dim=-1)
            return sample_tokens
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
