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
    
    
class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=15):
        super(UpSampleBlock, self).__init__()
        self.upsample = UpSampleLinear(in_channels)
        self.resconv = ConvResidual(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.upsample(x)
        x = self.resconv(x)
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
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask = None):
        assert q.shape[-1] == self.embed_dim, 'Input dimension mismatch, expected %d, got %d' % (self.embed_dim, q.shape[-1])
        assert k.shape[-1] == self.embed_dim, 'Input dimension mismatch, expected %d, got %d' % (self.embed_dim, k.shape[-1])
        assert v.shape[-1] == self.embed_dim, 'Input dimension mismatch, expected %d, got %d' % (self.embed_dim, v.shape[-1])
        # q, k, v are shape (b, n, d)
        
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)       
        # q, k, v are shape (b, n, h*d)
        
        q = rearrange(q, "b n (h d) -> b h n d", h = self.n_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h = self.n_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h = self.n_heads)
        # q, k, v are shape (b, h, n, d)
        
        W = einsum(q, k, "b h i d, b h j d -> b h i j") * self.scale
        # W is shape (b, h, n, n)
        
        if mask is not None:
            W = W.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(W, dim = -1)
        attn = self.dropout(attn)
        
        attn = einsum(attn, v, "b h i j, b h j d -> b h i d")
        # attn is shape (b, h, n, d)
        
        attn = rearrange(attn, "b h n d -> b n (h d)")
        # attn is shape (b, n, h*d)
        
        attn = self.w_o(attn)
        # attn is shape (b, n, d)
        
        return attn
    
    
class WindowAttentionBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, win_size, dropout=0.1):
        super(WindowAttentionBlock, self).__init__()
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.win_size = win_size
        self.scale = embed_dim ** -0.5
        
        self.w_q = nn.Linear(self.embed_dim, self.embed_dim * n_heads)
        self.w_k = nn.Linear(self.embed_dim, self.embed_dim * n_heads)
        self.w_v = nn.Linear(self.embed_dim, self.embed_dim * n_heads)
        self.w_o = nn.Linear(self.embed_dim * n_heads, self.embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q, k, v, mask = None):
        assert q.shape[-1] == self.embed_dim, 'Input dimension mismatch, expected %d, got %d' % (self.embed_dim, q.shape[-1])
        assert k.shape[-1] == self.embed_dim, 'Input dimension mismatch, expected %d, got %d' % (self.embed_dim, k.shape[-1])
        assert v.shape[-1] == self.embed_dim, 'Input dimension mismatch, expected %d, got %d' % (self.embed_dim, v.shape[-1])

        b, N, D = q.shape
        # q, k, v are shape (b, N, D), N = n * w, D = d * h
                
        q = rearrange(q, 'b (w n) D-> b w n D', w = self.win_size)
        k = rearrange(k, 'b (w n) D-> b w n D', w = self.win_size)
        v = rearrange(v, 'b (w n) D-> b w n D', w = self.win_size)
        # q, k, v are shape (b, w, n, d)
        
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)
        # q, k, v are shape (b, w, n, h*d)
        
        q = rearrange(q, "b w n (h d) -> b w h n d", h = self.n_heads)
        k = rearrange(k, "b w n (h d) -> b w h n d", h = self.n_heads)
        v = rearrange(v, "b w n (h d) -> b w h n d", h = self.n_heads)
        
        W = einsum(q, k, "b w h i d, b w h j d -> b w h i j") * self.scale
        # W is shape (b, w, h, n, n)
        
        if mask is not None:
            W = W.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(W, dim = -1)
        attn = self.dropout(attn)
        
        attn = einsum(attn, v, "b w h i j, b w h j d -> b w h i d")
        # attn is shape (b, w, h, n, d)
        
        attn = rearrange(attn, "b w h n d -> b w n (h d)")
        # attn is shape (b, w, n, h*d)
        
        attn = self.w_o(attn)
        # attn is shape (b, w, n, D)
        
        attn = rearrange(attn, "b w n d -> b (w n) d")
        # attn is shape (b, N, D)
        
        return attn


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout = 0.1, window_size = None):
        super(TransformerBlock, self).__init__()
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.window_size = window_size
        
        if window_size is None:
            self.attBlock = AttentionBlock(embed_dim, n_heads, dropout)
        else:
            self.attBlock = WindowAttentionBlock(embed_dim, n_heads, window_size, dropout)
        self.dropout = nn.Dropout(dropout)

        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            Residual(
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * n_heads),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(embed_dim * n_heads, embed_dim),
                    nn.Dropout(dropout),
                )
            ),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, q, k, v, mask=None):
        att = self.attBlock(q, k, v, mask)
        att = self.dropout(att)
        att = att + q
        return self.net(att)
    
class CrossAttentionWavUNet(nn.Module):
    def __init__(self, N_diffusion = 1000, conditional_dim = 32, out_dims = 1):
        super(CrossAttentionWavUNet, self).__init__()
        
        self.in_res_conv = ConvResidual(1, 4, 15)
        self.out_res_conv= ConvResidual(4, out_dims, 15)
        
        self.out_dims = out_dims
        
        # (4, 8000)
        self.DC1 = nn.Sequential(
            DownsampleBlock(4, 16),
            DownsampleBlock(16, 32)
        )
        # (32, 2000)
        self.DC2 = nn.Sequential(
            DownsampleBlock(32, 64, 9),
            DownsampleBlock(64, 128, 9)
        )
        # (128, 500)
        self.DC3 = DownsampleBlock(128, 256, 5)
        # (256, 250)
        
        self.selfAttention = TransformerBlock(256, 4)
        
        self.Windowed_Cross_Attention_3 = TransformerBlock(128, 4, 0.1, 50)
        self.Windowed_Cross_Attention_2 = TransformerBlock(32, 4, 0.1, 200)
        self.Windowed_Cross_Attention_1 = TransformerBlock(4, 4, 0.1, 400)
        
        if conditional_dim is not None:
            self.c_to_1 = nn.Linear(conditional_dim, 4)
            self.c_to_2 = nn.Linear(conditional_dim, 32)
            self.c_to_3 = nn.Linear(conditional_dim, 128)
            self.c_to_self = nn.Linear(conditional_dim, 256)
        
        self.US3 = UpSampleBlock(256, 128, 5)
        
        self.US2 = nn.Sequential(
            UpSampleBlock(128, 64, 9),
            UpSampleBlock(64, 32, 9)
        )
        
        self.US1 = nn.Sequential(
            UpSampleBlock(32, 16), 
            UpSampleBlock(16, 4)
        )
        
        self.waveform_time_embedding_q_1 = nn.Parameter(torch.randn(8000, 4))
        self.waveform_time_embedding_q_2 = nn.Parameter(torch.randn(2000, 32))
        self.waveform_time_embedding_q_3 = nn.Parameter(torch.randn(500, 128))
        
        self.waveform_time_embedding_q_self = nn.Parameter(torch.randn(250, 256))
        
        self.waveform_time_embedding_kv_1 = nn.Parameter(torch.randn(8000, 4))
        self.waveform_time_embedding_kv_2 = nn.Parameter(torch.randn(2000, 32))
        self.waveform_time_embedding_kv_3 = nn.Parameter(torch.randn(500, 128))
        
        self.diffusion_time_embedding_1 = nn.Embedding(N_diffusion, 4)
        self.diffusion_time_embedding_2 = nn.Embedding(N_diffusion, 32)
        self.diffusion_time_embedding_3 = nn.Embedding(N_diffusion, 128)
        
        self.diffusion_time_embedding_self = nn.Embedding(N_diffusion, 256)
    
    
    def forward(self, x, t, c = None):
        # x: (batch_size, 1, 8000)
        # t: (batch_size, 1)
        
        if c is not None:
            c1 = self.c_to_1(c)
            c2 = self.c_to_2(c)
            c3 = self.c_to_3(c)
            c_to_self = self.c_to_self(c)
        else:
            c1 = 0
            c2 = 0
            c3 = 0
            c_to_self = 0
        
        in_res = self.in_res_conv(x)
        dc1 = self.DC1(in_res)
        dc2 = self.DC2(dc1)
        dc3 = self.DC3(dc2)
        
        dc3 = rearrange(dc3, 'b c n -> b n c')
        dc3 = dc3 + self.diffusion_time_embedding_self(t) + self.waveform_time_embedding_q_self + c_to_self
        sa = self.selfAttention(dc3, dc3, dc3)
        sa = rearrange(sa, 'b n c -> b c n')
        
        us3 = self.US3(sa)
        us3 = rearrange(us3, 'b c n -> b n c')
        dc2 = rearrange(dc2, 'b c n -> b n c')
        us3 = us3 + self.diffusion_time_embedding_3(t) + self.waveform_time_embedding_q_3 + c3
        dc2 = dc2 + self.diffusion_time_embedding_3(t) + self.waveform_time_embedding_kv_3 + c3
        xa3 = self.Windowed_Cross_Attention_3(us3, dc2, dc2)
        xa3 = rearrange(xa3, 'b n c -> b c n')
        
        us2 = self.US2(xa3)
        us2 = rearrange(us2, 'b c n -> b n c')
        dc1 = rearrange(dc1, 'b c n -> b n c')
        us2 = us2 + self.diffusion_time_embedding_2(t) + self.waveform_time_embedding_q_2 + c2
        dc1 = dc1 + self.diffusion_time_embedding_2(t) + self.waveform_time_embedding_kv_2 + c2
        xa2 = self.Windowed_Cross_Attention_2(us2, dc1, dc1)
        xa2 = rearrange(xa2, 'b n c -> b c n')
        
        us1 = self.US1(xa2)
        us1 = rearrange(us1, 'b c n -> b n c')
        in_res = rearrange(in_res, 'b c n -> b n c')
        us1 = us1 + self.diffusion_time_embedding_1(t) + self.waveform_time_embedding_q_1 + c1
        in_res = in_res + self.diffusion_time_embedding_1(t) + self.waveform_time_embedding_kv_1 + c1
        xa1 = self.Windowed_Cross_Attention_1(us1, in_res, in_res)
        xa1 = rearrange(xa1, 'b n c -> b c n')
        
        out_res = self.out_res_conv(xa1)
        
        return out_res