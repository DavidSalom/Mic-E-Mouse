"""
MicEMouse Utils
"""
import torch.nn.functional as F
import torchaudio
import torch
from einops import reduce, rearrange
import einops

def trim_or_pad(x, T):
    if x.shape[-1] > T:
        return x[:T]
    else:
        # return F.pad(x, (0, T - x.shape[-1]))
        # pad at the end instead
        return F.pad(x, (0, T - x.shape[-1]), mode='constant', value=0)

def trim_or_pad2(x, T):
    if x.shape[-1] > T:
        return x[:, :T]
    else:
        return F.pad(x, (0, T - x.shape[-1]), mode="constant", value=0)

def buildTransforms(device, Fs, n_fft, win_length, hop_length, n_mels, f_max_mouse, f_max_full):
    specFn = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        power=None,
    ).to(device)

    inverseSpecFn = torchaudio.transforms.GriffinLim(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        power=2,
    ).to(device)

    melFilterMouse = torchaudio.functional.melscale_fbanks(
        n_mels=n_mels,
        sample_rate=Fs,
        f_min=20,
        f_max=f_max_mouse,
        n_freqs=n_fft// 2 + 1,
    ).T.to(device)

    melFilter = torchaudio.functional.melscale_fbanks(
        n_mels=n_mels,
        sample_rate=Fs,
        f_min=20,
        f_max=f_max_full,
        n_freqs=n_fft// 2 + 1,
    ).T.to(device)

    inverseMelFn = torchaudio.transforms.InverseMelScale(
        n_stft=n_fft//2 + 1,
        n_mels=n_mels,
        sample_rate=Fs,
        f_min=20,
        f_max=f_max_full,
    ).to(device)
    return specFn, inverseSpecFn, melFilterMouse, melFilter, inverseMelFn

def normalize(batch):
    M, W = batch
    # Zero mean
    W -= reduce(W, "b t -> b ()", "mean")
    M -= reduce(M, "b c t -> b c ()", "mean")
    # # Normalize power
    W /= torch.std(W)
    M /= torch.std(M)
    return M, W

def denormalize(X, Yorig):
    X *= torch.std(Yorig - reduce(Yorig, "b t -> b ()", "mean"))
    X += reduce(Yorig, "b t -> b ()", "mean")
    # # Normalize power
    return X

def toMelDB(batch, specFn, mousemelfilters, fullmelfilter):
    M, W = batch

    W = specFn(W)
    M = specFn(M)

    Wphase = W.angle()
    Mphase = M.angle()

    W = W.abs().pow(2)
    M = M.abs().pow(2)


    W = torch.einsum("bct,fc->bft", W, fullmelfilter)
    M = torch.einsum("bkct,fc->bkft", M, mousemelfilters)
    
    M = rearrange(M, "b c m t -> b (m c) t", c=2)

    # Amplitude to dB (Whisper formula)
    W = (torch.maximum(torch.clamp(W, min=1e-10).log10(), torch.clamp(W, min=1e-10).log10().max() - 8.0) + 4.0)/ 4.0
    M = (torch.maximum(torch.clamp(M, min=1e-10).log10(), torch.clamp(M, min=1e-10).log10().max() - 8.0) + 4.0)/ 4.0

    return M, W, Mphase, Wphase

def mapToBounds(batch, mnmn, mxmx): # These funcs take two specs in batch (M, W), although the same operation is being done on both. From a software engineering perspective, this is a bit of a code smell, but it's necessary to keep the notebook code clean.
    M, W = batch
    # remap to [-1, 1]
    W = (W - mnmn) / (mxmx - mnmn)
    M = (M - mnmn) / (mxmx - mnmn)
    W = torch.clamp(W, 0, 1)
    M = torch.clamp(M, 0, 1)
    W = 2 * W - 1
    M = 2 * M - 1
    return  M, W

def inverseBounds(X, mnmn, mxmx):
    X = (X + 1) / 2
    X = X * (mxmx - mnmn) + mnmn
    return X

def fromMelDB(recon_batch, inverseMelFn, inverseSpecFn):
    recon_batch = recon_batch * 4 - 4
    recon_batch = torch.pow(10, recon_batch)
    recon_batch = inverseMelFn(recon_batch)
    recon_batch = inverseSpecFn(recon_batch)
    return recon_batch

def apply_wiener(X, filt):
    Xfft = torch.fft.rfft(X, dim=-1)
    Yfft = Xfft * filt
    return torch.fft.irfft(Yfft, dim=-1)