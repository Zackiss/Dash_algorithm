from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DotAttention(nn.Module, ABC):  # Compute scaled dot product attention.
    def __init__(self, ):
        super(DotAttention, self).__init__()

    @staticmethod
    def forward(query, key, value, mask, dropout):
        # query: [B, NH, L, D], key: [B, NH, L, D], value: [B, NH, L, D]
        scores = torch.matmul(query, key.transpose(-2, -1))  # [B, NH, L, L]
        scores = scores / math.sqrt(query.size(-1))

        scores = scores.masked_fill(mask, -1e9)

        prob_attn = F.softmax(scores, dim=-1)  # [B, NH, L, L]
        prob_attn = dropout(prob_attn)
        return torch.matmul(prob_attn, value), prob_attn  # [B, NH, L, D], [B, NH, L, L]


class GeneralAttention(nn.Module, ABC):
    def __init__(self, head_size):
        super(GeneralAttention, self).__init__()
        self.linear = nn.Linear(head_size, head_size, bias=False)
        # self.l1 = nn.Parameter(torch.rand(1))

    def forward(self, query, key, value, mask, dropout):
        # query: [B, NH, L, D], key: [B, NH, L, D], value: [B, NH, L, D]
        scores = torch.matmul(query, self.linear(key).transpose(-2, -1))  # [B, NH, L, L]
        scores = scores / math.sqrt(query.size(-1))

        scores = scores.masked_fill(mask, -1e9)
        prob_attn = F.softmax(scores, dim=-1)  # [B, NH, L, L]

        # prob_attn = dropout(prob_attn)
        return torch.matmul(prob_attn, value), prob_attn  # [B, NH, L, D], [B, NH, L, L]
