import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    """Single Head of Attention"""
    def __init__(self, n_embd, block_size, head_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias= False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        wei =  q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim = -1)
        wei = self.dropout(wei)

        val = self.value(x)
        out = wei @ val
        return out
    
class MultiHeadAttention(nn.Module):
    """MultiHead Attention using single head above"""
    def __init__(self, n_embd, block_size, head_size, num_heads, dropout):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(n_embd = n_embd, block_size = block_size, head_size = head_size, dropout = dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out =  torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.proj(out)
        out = self.dropout(out)
        return out