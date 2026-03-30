
import torch
import torch.nn as nn
from core.nn.whisperPrimitives import TransformerChain

class ReconstructionModel(nn.Module):
    def __init__(self, n_mels=80, embed_dim=384, n_head=6, n_layer=4, kernel_size=15):
        super().__init__()
        self.n_mels = n_mels
        self.embed_dim = embed_dim
        padding = kernel_size // 2
        
        self.model = nn.Sequential(
            nn.Conv1d(n_mels * 2, embed_dim, kernel_size, padding=padding),
            nn.GELU(),
            TransformerChain(n_ctx=501, n_state=embed_dim, n_head=n_head, n_layer=n_layer),
            nn.GELU(),
            nn.Conv1d(embed_dim, n_mels * 4, kernel_size, padding=padding),
            nn.GELU(),
            nn.Conv1d(n_mels * 4, n_mels, 1, padding=0),
        )

    def forward(self, x):
        return self.model(x)
