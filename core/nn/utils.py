"""
MicEMouse Utils
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from einops import reduce, rearrange
from tqdm import tqdm
from typing import Tuple, List
from .whisperPrimitives import AudioEncoder  # Import specifically needed classes or functions.

def trim_or_pad(x: torch.Tensor, T: int) -> torch.Tensor:
    """
    Trim or pad a 1D tensor to length T.
    
    :param x: Input tensor of shape (N,).
    :param T: Desired length.
    :return: Tensor of shape (T,).
    """
    if x.shape[-1] > T:
        return x[..., :T]
    else:
        padding = T - x.shape[-1]
        return F.pad(x, (0, padding), mode='constant', value=0)

def trim_or_pad2(x: torch.Tensor, T: int) -> torch.Tensor:
    """
    Trim or pad a 2D tensor along the last dimension to length T.
    
    :param x: Input tensor of shape (C, N).
    :param T: Desired length.
    :return: Tensor of shape (C, T).
    """
    if x.shape[-1] > T:
        return x[..., :T]
    else:
        padding = T - x.shape[-1]
        return F.pad(x, (0, padding), mode="constant", value=0)

class TransformWrapper:
    """
    A wrapper class for transforming audio signals to and from Mel spectrogram representations.
    """
    def __init__(
        self,
        Fs: int,
        n_fft: int,
        win_length: int,
        hop_length: int,
        n_mels: int,
        f_max_mouse: float,
        f_max_full: float
    ):
        # Spectrogram and inverse spectrogram functions.
        self.spec_fn = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            power=2,
        )
        self.inverse_spec_fn = torchaudio.transforms.GriffinLim(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            power=2,
        )

        # Mel filter banks for mouse and full frequency ranges.
        self.mel_filter_mouse = torchaudio.functional.melscale_fbanks(
            n_mels=n_mels,
            sample_rate=Fs,
            f_min=20,
            f_max=f_max_mouse,
            n_freqs=n_fft // 2 + 1,
        ).T
        self.mel_filter_full = torchaudio.functional.melscale_fbanks(
            n_mels=n_mels,
            sample_rate=Fs,
            f_min=20,
            f_max=f_max_full,
            n_freqs=n_fft // 2 + 1,
        ).T

        # Inverse Mel scale function.
        self.inverse_mel_fn = torchaudio.transforms.InverseMelScale(
            n_stft=n_fft // 2 + 1,
            n_mels=n_mels,
            sample_rate=Fs,
            f_min=20,
            f_max=f_max_full,
        )

        # Initialize min and max values for normalization.
        self.mnmn = None
        self.mxmx = None

    def to_mel_db(self, M: torch.Tensor, W: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert input signals to Mel-scaled decibel spectrograms.

        :param M: Mouse audio tensor of shape (B, C, T).
        :param W: Original audio tensor of shape (B, T).
        :return: Tuple of (M_mel_db, W_mel_db, M_phase, W_phase).
        """
        # Compute spectrograms.
        W_spec = self.spec_fn(W)
        M_spec = self.spec_fn(M)

        # Extract phase information.
        W_phase = W_spec.angle()
        M_phase = M_spec.angle()

        # Compute power spectra.
        W_power = W_spec.abs().pow(2)
        M_power = M_spec.abs().pow(2)

        # Apply Mel filter banks.
        W_mel = torch.matmul(W_power.transpose(1, 2), self.mel_filter_full).transpose(1, 2)
        M_mel = torch.matmul(M_power.view(M_power.size(0), -1, M_power.size(-1)).transpose(1, 2), self.mel_filter_mouse).transpose(1, 2)
        M_mel = M_mel.view(M_power.size(0), M_power.size(1), -1, M_mel.size(-1))  # Reshape back to (B, C, n_mels, T)

        # Convert amplitude to decibels.
        W_mel_db = self.amplitude_to_db(W_mel)
        M_mel_db = self.amplitude_to_db(M_mel)

        return M_mel_db, W_mel_db, M_phase, W_phase

    def amplitude_to_db(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert amplitude to decibel scale using a modified Whisper formula.

        :param x: Input tensor.
        :return: Tensor in decibel scale.
        """
        x = torch.clamp(x, min=1e-10)
        x_db = x.log10()
        x_db = (torch.maximum(x_db, x_db.max() - 8.0) + 4.0) / 4.0
        return x_db

    def from_mel_db(self, recon_batch: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct waveform from Mel-scaled decibel spectrogram.

        :param recon_batch: Input Mel spectrogram.
        :return: Reconstructed waveform.
        """
        recon_batch = recon_batch * 4 - 4  # Invert scaling
        recon_batch = torch.pow(10, recon_batch)
        recon_batch = self.inverse_mel_fn(recon_batch)
        recon_waveform = self.inverse_spec_fn(recon_batch)
        return recon_waveform

    def compute_unbounded_spec(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute the unbounded spectrogram of the input signal.

        :param X: Input audio tensor.
        :return: Spectrogram tensor.
        """
        ms = True if X.dim() == 3 and X.size(1) == 2 else False

        # Compute spectrogram.
        X_spec = self.spec_fn(X)

        if ms:
            # For multi-channel input.
            X_mel = torch.matmul(
                X_spec.view(X_spec.size(0), -1, X_spec.size(-1)).transpose(1, 2),
                self.mel_filter_mouse
            ).transpose(1, 2)
            X_mel = X_mel.view(X_spec.size(0), X_spec.size(1), -1, X_mel.size(-1))
            X_mel = rearrange(X_mel, "b c m t -> b (m c) t")
        else:
            # For single-channel input.
            X_mel = torch.matmul(X_spec.transpose(1, 2), self.mel_filter_full).transpose(1, 2)

        # Convert amplitude to decibels.
        X_mel_db = self.amplitude_to_db(X_mel)
        return X_mel_db

    def compute_min_max(self, loader) -> None:
        """
        Compute the minimum and maximum values across the dataset for normalization.

        :param loader: DataLoader providing the dataset.
        """
        mnmn = float('inf')
        mxmx = float('-inf')
        for Mwav, Wwav, _, _ in loader:
            M = self.compute_unbounded_spec(Mwav)
            W = self.compute_unbounded_spec(Wwav)
            mnmn = min(mnmn, M.min().item(), W.min().item())
            mxmx = max(mxmx, M.max().item(), W.max().item())
        self.mnmn = mnmn
        self.mxmx = mxmx

    def compute_spec(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute the normalized spectrogram of the input signal.

        :param X: Input audio tensor.
        :return: Normalized spectrogram tensor.
        """
        X_mel_db = self.compute_unbounded_spec(X)
        if self.mnmn is not None and self.mxmx is not None:
            # Normalize to [-1, 1]
            X_norm = 2 * (X_mel_db - self.mnmn) / (self.mxmx - self.mnmn) - 1
            X_norm = torch.clamp(X_norm, -1, 1)
            return X_norm
        else:
            raise ValueError("Min and max values not computed. Run 'compute_min_max' first.")

    def from_spec(self, X: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct waveform from normalized spectrogram.

        :param X: Normalized spectrogram tensor.
        :return: Reconstructed waveform.
        """
        if self.mnmn is None or self.mxmx is None:
            raise ValueError("Min and max values not set. Cannot invert normalization.")

        # Invert normalization
        X = (X + 1) / 2
        X = X * (self.mxmx - self.mnmn) + self.mnmn
        X = 10 ** (4 * X - 4)
        X = self.inverse_mel_fn(X)
        recon_waveform = self.inverse_spec_fn(X)
        return recon_waveform

    def to(self, device: torch.device):
        """
        Move all internal tensors and models to the specified device.

        :param device: Target device ('cpu' or 'cuda').
        """
        self.spec_fn = self.spec_fn.to(device)
        self.inverse_spec_fn = self.inverse_spec_fn.to(device)
        self.mel_filter_mouse = self.mel_filter_mouse.to(device)
        self.mel_filter_full = self.mel_filter_full.to(device)
        self.inverse_mel_fn = self.inverse_mel_fn.to(device)
        return self

def normalize(batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize a batch of audio data.

    :param batch: Tuple of (M, W) audio tensors.
    :return: Normalized (M, W) tensors.
    """
    M, W = batch
    # Zero mean
    W = W - W.mean(dim=-1, keepdim=True)
    M = M - M.mean(dim=-1, keepdim=True)
    # Normalize standard deviation
    W = W / W.std(dim=-1, keepdim=True)
    M = M / M.std(dim=-1, keepdim=True)
    return M, W

def denormalize(X: torch.Tensor, Y_orig: torch.Tensor) -> torch.Tensor:
    """
    Denormalize the tensor X using the statistics of Y_orig.

    :param X: Normalized tensor.
    :param Y_orig: Original tensor used for computing statistics.
    :return: Denormalized tensor.
    """
    std = Y_orig.std(dim=-1, keepdim=True)
    mean = Y_orig.mean(dim=-1, keepdim=True)
    X = X * std + mean
    return X

def map_to_bounds(batch: Tuple[torch.Tensor, torch.Tensor], mnmn: float, mxmx: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Map batch data to the range [-1, 1].

    :param batch: Tuple of (M, W) tensors.
    :param mnmn: Minimum value for normalization.
    :param mxmx: Maximum value for normalization.
    :return: Tuple of normalized (M, W) tensors.
    """
    M, W = batch
    W = (W - mnmn) / (mxmx - mnmn)
    M = (M - mnmn) / (mxmx - mnmn)
    W = torch.clamp(W, 0, 1)
    M = torch.clamp(M, 0, 1)
    W = 2 * W - 1
    M = 2 * M - 1
    return M, W

def inverse_bounds(X: torch.Tensor, mnmn: float, mxmx: float) -> torch.Tensor:
    """
    Invert the mapping from [-1, 1] back to the original scale.

    :param X: Normalized tensor.
    :param mnmn: Minimum value used during normalization.
    :param mxmx: Maximum value used during normalization.
    :return: Tensor in the original scale.
    """
    X = (X + 1) / 2
    X = X * (mxmx - mnmn) + mnmn
    return X

def apply_wiener(X: torch.Tensor, filt: torch.Tensor) -> torch.Tensor:
    """
    Apply a Wiener filter to the input signal.

    :param X: Input signal tensor.
    :param filt: Filter tensor.
    :return: Filtered signal tensor.
    """
    X_fft = torch.fft.rfft(X, dim=-1)
    Y_fft = X_fft * filt
    return torch.fft.irfft(Y_fft, dim=-1)

class Permute(nn.Module):
    """
    A module to permute tensor dimensions.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 1)

def build_classifier(
    n_mels: int,
    n_classes: int,
    time_len: int,
    inner_dim: int,
    device: torch.device
) -> nn.Module:
    """
    Build a classifier model.

    :param n_mels: Number of Mel frequency bins.
    :param n_classes: Number of output classes.
    :param time_len: Length of the time dimension.
    :param inner_dim: Inner dimension size for linear layers.
    :param device: Device to place the model on.
    :return: The classifier model.
    """
    model = nn.Sequential(
        nn.Conv1d(n_mels, 128, kernel_size=3),
        nn.GELU(),
        nn.Conv1d(128, 128, kernel_size=3, stride=2),
        AudioEncoder(n_mels=64, n_ctx=(time_len // 2 - 1), n_state=128, n_head=8, n_layer=4),
        Permute(),
        nn.GELU(),
        nn.Conv1d(128, 64, kernel_size=3, stride=2),
        nn.GELU(),
        nn.Conv1d(64, 32, kernel_size=3, stride=2),
        nn.GELU(),
        nn.Flatten(),
        nn.Linear(inner_dim, 128),
        nn.ReLU(),
        nn.Linear(128, n_classes),
    ).to(device)
    return model

def train(
    model: nn.Module,
    src: str,
    tgt: str,
    loader: torch.utils.data.DataLoader,
    list_of_labels: List,
    tf_wrapper: TransformWrapper,
    device: torch.device,
    n_epochs: int = 10
):
    """
    Train the model.

    :param model: The classifier model.
    :param src: Source data key ('MS' or other).
    :param tgt: Target label key ('SPEAKER' or other).
    :param loader: DataLoader providing the training data.
    :param list_of_labels: List of possible label values.
    :param tf_wrapper: Instance of TransformWrapper for processing.
    :param device: Device to perform training on.
    :param n_epochs: Number of epochs to train.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    total_steps = n_epochs * len(loader)

    with tqdm(total=total_steps) as pbar:
        for epoch in range(n_epochs):
            for batch in loader:
                X1, X2, Y1, Y2 = batch
                X = X1 if src == "MS" else X2
                Y = Y1 if tgt == "SPEAKER" else Y2

                loss = update(model, optimizer, criterion, X, Y, list_of_labels, tf_wrapper, device)
                pbar.set_description(f"Epoch {epoch+1}/{n_epochs} - Loss: {loss:.4f}")
                pbar.update(1)

def update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    X: torch.Tensor,
    Y: List,
    class_list: List,
    tf_wrapper: TransformWrapper,
    device: torch.device
) -> float:
    """
    Perform a single update step.

    :param model: The classifier model.
    :param optimizer: Optimizer instance.
    :param criterion: Loss function.
    :param X: Input data tensor.
    :param Y: Target labels.
    :param class_list: List of class labels.
    :param tf_wrapper: TransformWrapper instance.
    :param device: Device to perform computation on.
    :return: Loss value.
    """
    X = X.to(device)
    X = tf_wrapper.compute_spec(X)

    Y_indices = torch.tensor([class_list.index(i) for i in Y], device=device)
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, Y_indices)
    loss.backward()
    optimizer.step()
    return loss.item()

def get_accuracy(
    model: nn.Module,
    src: str,
    tgt: str,
    loader: torch.utils.data.DataLoader,
    class_list: List,
    tf_wrapper: TransformWrapper,
    device: torch.device
) -> float:
    """
    Calculate the accuracy of the model.

    :param model: The classifier model.
    :param src: Source data key ('MS' or other).
    :param tgt: Target label key ('SPEAKER' or other).
    :param loader: DataLoader providing the evaluation data.
    :param class_list: List of class labels.
    :param tf_wrapper: TransformWrapper instance.
    :param device: Device to perform computation on.
    :return: Accuracy as a float between 0 and 1.
    """
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            X1, X2, Y1, Y2 = batch
            X = X1 if src == "MS" else X2
            Y = Y1 if tgt == "SPEAKER" else Y2

            X = X.to(device)
            X = tf_wrapper.compute_spec(X)
            Y_indices = torch.tensor([class_list.index(i) for i in Y], device=device)

            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            total += Y_indices.size(0)
            correct += (predicted == Y_indices).sum().item()
    model.train()
    return correct / total
