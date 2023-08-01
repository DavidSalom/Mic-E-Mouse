import torch
import torch.nn as nn
import torch.nn.functional as F
from .NetworkPrimitives import Residual
from torch.nn import BatchNorm1d, Conv1d, MaxPool1d
import math
from einops import rearrange, repeat, reduce, einsum

class ConvResidual(nn.Module):
    """
    in -> Residual(conv, bn, gelu, conv), bn, conv, gelu, bn -> out
    """
    def __init__(self, in_channels, out_channels, kernel_size = 15):
        super(ConvResidual, self).__init__()
        self.net = nn.Sequential(
            Residual(
                nn.Sequential(
                    nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding='same'),
                    BatchNorm1d(in_channels),
                    nn.GELU(),
                    nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding='same')
                )
            ),
            BatchNorm1d(in_channels),
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding='same'),
            nn.GELU(),
            BatchNorm1d(out_channels)
        )
    
    def forward(self, x):
        return self.net(x)

class DownSample(nn.Module):
    def __init__(self):
        super(DownSample, self).__init__()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.maxpool(x)
        return x
    
class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=15):
        super(DownsampleBlock, self).__init__()
        self.resconv = ConvResidual(in_channels, out_channels, kernel_size)
        self.downsample = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.resconv(x)
        x = self.downsample(x)
        return x


class UpSampleLinear(nn.Module):
    def __init__(self, n_channels):
        super(UpSampleLinear, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.conv = nn.Conv1d(n_channels, n_channels, kernel_size=5, stride=1, padding='same')
        self.bn = BatchNorm1d(n_channels)
        
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(F.gelu(x))
        x = self.bn(x)

        return x
    

class CropConcat1d(nn.Module):
    def __init__(self):
        """
        Crop and concatenate two 1d tensors of different sizes, assuming the first tensor is larger than the second one
        """
        super(CropConcat1d, self).__init__()

    def forward(self, x1, x2):
        center = x1.size(2) // 2
        half = x2.size(2) // 2
        x1 = x1[:, :, center-half:center+half]
        x = torch.cat([x1, x2], dim=1)
        return x
        

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.register_buffer('pe', self._get_sinusoid_encoding_table(d_model, max_len))
        
    def _get_sinusoid_encoding_table(self, d_model, max_len):
        """Sinusoid position encoding table"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position.float() * div_term.float())
        pe[:, 1::2] = torch.cos(position.float() * div_term.float())
        return pe.unsqueeze(0)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(LearnedPositionalEncoding, self).__init__()
        self.pe = nn.Embedding(max_len, d_model)
        
    def forward(self, x):
        return self.pe(torch.arange(x.size(1)).to(x.device)).unsqueeze(0)

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout=0.1):
        super(AttentionBlock, self).__init__()
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.scale = embed_dim ** -0.5
        
        self.w_q = nn.Linear(self.embed_dim, self.embed_dim * n_heads)
        self.w_k = nn.Linear(self.embed_dim, self.embed_dim * n_heads)
        self.w_v = nn.Linear(self.embed_dim, self.embed_dim * n_heads)
        self.w_o = nn.Linear(self.embed_dim * n_heads, self.embed_dim)
        
    def forward(self, q, k, v, mask = None):
        assert q.shape[-1] == self.embed_dim, 'Input dimension mismatch, expected %d, got %d' % (self.embed_dim, q.shape[-1])
        assert k.shape[-1] == self.embed_dim, 'Input dimension mismatch, expected %d, got %d' % (self.embed_dim, k.shape[-1])
        assert v.shape[-1] == self.embed_dim, 'Input dimension mismatch, expected %d, got %d' % (self.embed_dim, v.shape[-1])

        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)       
        
        W = einsum("... i d, ... j d -> ... i j", q, k) * self.scale
        attn = W.softmax(dim=-1)
        
        attn = self.w_o(attn)
        
        return attn

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout = 0.1):
        super(TransformerBlock, self).__init__()

        self.ln1 = nn.LayerNorm(embed_dim)
        self.attention = AttentionBlock(embed_dim, n_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        
        self.ln2 = nn.LayerNorm(embed_dim)
        
        self.fc1 = nn.Linear(embed_dim, embed_dim * n_heads)
        self.fc2 = nn.Linear(embed_dim * n_heads, embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, y):
        att = self.ln1(x)
        att = self.attention(att, y, y)
        att = self.dropout1(att)
        x = x + att
        att = self.ln2(x)
        att = self.fc1(att)
        att = F.gelu(att)
        att = self.fc2(att)
        att = self.dropout2(att)
        x = x + att
        return x