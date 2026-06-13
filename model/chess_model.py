import torch
import torch.nn as nn
from torch.nn import functional as F
from transformer import Block

class ChessEvaluator:
    def __init__(self, d_model, num_heads, num_layers):
        super().__init__()
        
        # input projetion (12 piece channel -> d_model)
        self.piece_embedding = nn.Linear(12, d_model)

        # positional embedding, 8x8, 64 squares
        self.pos_embedding = nn.Embedding(64, d_model)

        # transformer blocks
        self.blocks = nn.Sequential(*[Block(d_model, num_heads) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(d_model)

        # regression head
        self.regression_head = nn.Linear(d_model, 1)
    
    def forward(self, x):
        # x is in shape (Batch, 12, 8, 8)
        B, C, H, W = x.shape

        # flattening the spatial grid to (Batch, 12, 64)
        x = x.view(B, C, H * W)

        # swap axes to get (Batch, 64, 12)
        x = x.transpose(1,2)

        # forward pass
        # embedding
        x = self.piece_embedding(x)
        positions = torch.arange(0, 64, device = x.device).unsqueeze(0) # shape: (1, 64)
        x = x + self.pos_embedding(positions)

        # transformer, pass through blocks
        x = self.blocks(x)
        x = self.ln_f(x)

        # regression head
        x = x.mean(dim=1)
        
        logits = self.regression_head(x)
        win_prob = torch.sigmoid(logits)

        return win_prob

    
