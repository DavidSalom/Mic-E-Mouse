"""
MicEMouse Utils
"""
import torch.nn.functional as F
import torchaudio
import torch
from einops import reduce, rearrange
import einops
from tqdm import tqdm
import torch.nn as nn
from .whisperPrimitives import *

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

class transformWrapper():
    def __init__(self, Fs, n_fft, win_length, hop_length, n_mels, f_max_mouse, f_max_full):
        self.specFn = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            power=2,
        )
        self.inverseSpecFn = torchaudio.transforms.GriffinLim(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            power=2,
        )
        self.melFilterMouse = torchaudio.functional.melscale_fbanks(
            n_mels=n_mels,
            sample_rate=Fs,
            f_min=20,
            f_max=f_max_mouse,
            n_freqs=n_fft// 2 + 1,
        ).T
        self.melFilter = torchaudio.functional.melscale_fbanks(
            n_mels=n_mels,
            sample_rate=Fs,
            f_min=20,
            f_max=f_max_full,
            n_freqs=n_fft// 2 + 1,
        ).T

        self.inverseMelFn = torchaudio.transforms.InverseMelScale(
            n_stft=n_fft//2 + 1,
            n_mels=n_mels,
            sample_rate=Fs,
            f_min=20,
            f_max=f_max_full,
        )
    
    def toMelDB(self, batch):
        M, W = batch

        W = self.specFn(W)
        M = self.specFn(M)

        Wphase = W.angle()
        Mphase = M.angle()

        W = W.abs().pow(2)
        M = M.abs().pow(2)


        W = torch.einsum("bct,fc->bft", W, self.fullmelfilter)
        M = torch.einsum("bkct,fc->bkft", M, self.mousemelfilters)
        
        M = rearrange(M, "b c m t -> b (m c) t", c=2)

        # Amplitude to dB (Whisper formula)
        W = (torch.maximum(torch.clamp(W, min=1e-10).log10(), torch.clamp(W, min=1e-10).log10().max() - 8.0) + 4.0)/ 4.0
        M = (torch.maximum(torch.clamp(M, min=1e-10).log10(), torch.clamp(M, min=1e-10).log10().max() - 8.0) + 4.0)/ 4.0

        return M, W, Mphase, Wphase
    
    def fromMelDB(self, recon_batch):
        recon_batch = recon_batch * 4 - 4
        recon_batch = torch.pow(10, recon_batch)
        recon_batch = self.inverseMelFn(recon_batch)
        recon_batch = self.inverseSpecFn(recon_batch)
        return recon_batch

    def computeUnboundedSpec(self, X):
        ms = True if X.shape[1] == 2 else False
        # if ms:
        #     X -= einops.reduce(X, "b c t -> b c ()", "mean")  # Remove mean
        # else:
        #     X -= einops.reduce(X, "b t -> b ()", "mean")  # Remove mean
        # X /= torch.std(X)                                 # Normalize                                        
        X = self.specFn(X)                                # Convert to mel
        # X = X.abs().pow(2)                                # Power
        if ms:
            X = torch.einsum("bkct,fc->bkft", X, self.melFilterMouse)
            X = einops.rearrange(X, "b c m t -> b (m c) t", c=2)
        else:
            X = torch.einsum("bct,fc->bft", X, self.melFilter) # Apply mel filter
        X = (torch.maximum(torch.clamp(X, min=1e-10).log10(), torch.clamp(X, min=1e-10).log10().max() - 8.0) + 4.0)/ 4.0    # Convert to dB
        return X

    def computeMinMax(self, loader):
        return
        mnmn = torch.inf
        mxmx = -torch.inf
        for Mwav, Wwav, _, _ in loader:
            M = self.computeUnboundedSpec(Mwav)
            W = self.computeUnboundedSpec(Wwav)
            mnmn = min(mnmn, M.min(), W.min())
            mxmx = max(mxmx, M.max(), W.max())
        self.mnmn = mnmn
        self.mxmx = mxmx

    def computeSpec(self, X):
        X = self.computeUnboundedSpec(X)
        # X = (X - self.mnmn) / (self.mxmx - self.mnmn)               # Map to bounds
        # X = torch.clamp(X, 0, 1)                    # Clamp to [0, 1]
        # X = 2 * X - 1                               # Map to [-1, 1]
        return X
    
    def fromSpec(self, X):
        if X.shape[1] == 160:
            X = einops.rearrange(X, "b (m c) t -> b c m t", c=2)
            X = torch.mean(X, dim=1)
        X = (X + 1) / 2
        X = X * (self.mxmx - self.mnmn) + self.mnmn
        X = 10 ** (4 * X - 4)
        X = self.inverseMelFn(X)
        X = self.inverseSpecFn(X)
        return X

    def to(self, device):
        self.specFn = self.specFn.to(device)
        self.inverseSpecFn = self.inverseSpecFn.to(device)
        self.melFilterMouse = self.melFilterMouse.to(device)
        self.melFilter = self.melFilter.to(device)
        self.inverseMelFn = self.inverseMelFn.to(device)
        return self

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

def apply_wiener(X, filt):
    Xfft = torch.fft.rfft(X, dim=-1)
    Yfft = Xfft * filt
    return torch.fft.irfft(Yfft, dim=-1)

flatten = nn.Flatten()
class permute (nn.Module):
    def forward(self, x):
        return x.permute(0, 2, 1)

def train(model, src, tgt, loader, list_of_labels, tf_wrapper, device, n_epochs=10):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    with tqdm(total = n_epochs * len(loader)) as pbar:
        for epoch in range(n_epochs):
            for (X1, X2, Y1, Y2) in loader:
                if src == "MS":
                    X = X1
                else:
                    X = X2
                if tgt == "SPEAKER":
                    Y = Y1
                else:
                    Y = Y2
                    
                loss = update(model, optimizer, criterion, X, Y, list_of_labels, tf_wrapper, device)
                pbar.set_description(f"Loss: {loss:.4f}")
                pbar.update(1)
            

buildClassifier = lambda n_mels, n_classes, time_len, inner_dim, device: nn.Sequential(
    nn.Conv1d(n_mels, 128, 3),
    nn.GELU(),
    nn.Conv1d(128, 128, 3, stride = 2),
    AudioEncoder(n_mels=64, n_ctx=(time_len // 2 - 1), n_state=128, n_head = 8, n_layer=4),
    permute(),
    nn.GELU(),
    nn.Conv1d(128, 64, 3, stride = 2),
    nn.GELU(),
    nn.Conv1d(64, 32, 3, stride = 2),
    nn.GELU(),
    nn.Flatten(),
    nn.Linear(inner_dim, 128),
    nn.ReLU(),
    nn.Linear(128, n_classes),
).to(device)

def update(model, optimizer, criterion, X, Y, class_list, tf_wrapper, device):
    X = X.to(device)
    X = tf_wrapper.computeSpec(X)

    Y = torch.tensor([class_list.index(i) for i in Y], device=device)
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, Y)
    loss.backward()
    optimizer.step()
    return loss.item()

def getAccuracy(model, src, tgt, loader, class_list, tf_wrapper, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for (X1, X2, Y1, Y2) in loader:
            if src == "MS":
                X = X1
            else:
                X = X2
            if tgt == "SPEAKER":
                Y = Y1
            else:
                Y = Y2
            X = X.to(device)
            X = tf_wrapper.computeSpec(X)
            Y = torch.tensor([class_list.index(i) for i in Y], device=device)
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            total += Y.size(0)
            correct += (predicted == Y).sum().item()
    return correct / total