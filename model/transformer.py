"""
This module contains transformer architecture, mostly scrapped from my ml_practice
project following Karpathy's course with some tweaks.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    def __init__(self, n_embd, head_size):
        super().__init__()
        # K, Q, V Vectors
        self.key = nn.Linear(n_embd, head_size, bias = False)   # (B, T, head_size)
        self.query = nn.Linear(n_embd, head_size, bias = False) # (B, T, head_size)
        self.value = nn.Linear(n_embd, head_size, bias = False) # (B, T, head_size)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # attention scores
        affinities = q @ k.transpose(-2, -1)

        #scaling 
        logits = affinities / (k.shape[-1] ** 0.5)

        # activation func
        act = F.softmax(logits, dim=-1)
    
        out = act @ v # (B, T, head_size)

        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads):
        super().__init__()

        head_size = n_embd // num_heads
        self.heads = nn.ModuleList([Head(n_embd, head_size) for _ in range(num_heads)])
        self.linear = nn.Linear(num_heads * head_size, num_heads * head_size)

    def forward(self, x):
        # Calling forward pass for every head in list of heads
        out_list = [h(x) for h in self.heads]

        # concatenating the list of heads together
        cat = torch.cat(out_list, dim=-1)
        
        out = self.linear(cat)

        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        # Creating sequential network
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )
    def forward(self, x):
        # forward pass
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, num_heads):
        super().__init__()
        
        head_size = n_embd // num_heads

        self.multi = MultiHeadAttention(n_embd, num_heads)
        self.feed = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        # attention
        x = x + self.multi(self.ln1(x))

        # feed forward prediction
        x = x + self.feed(self.ln2(x))

        return x