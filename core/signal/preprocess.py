import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm
from typing import Tuple

def loadData(src: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load the data from the source file and return it as a PyTorch tensor.
    :param src: The path to the source file.
    :return: A Tuple containing the time vector, the nonuniform X vector, and the nonuniform Y vector.
    """
    # Load the data from the source file
    data = np.loadtxt(src, delimiter=',',skiprows=1,dtype=int)
    # Convert the data into a PyTorch tensor
    data = torch.from_numpy(data)

    # Extract the data from the PyTorch tensor
    T = data[:, 0]
    nonuniformX = data[:, 1]
    nonuniformY = data[:, 2]
    
    return T, nonuniformX, nonuniformY

def resample(T : torch.Tensor, nuX : torch.Tensor, nuY : torch.Tensor, Fs : int = 16000, resampleFn : str = "cubic", maxTimesteps : int = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Resample the signal to uniformize the sampling rate through interpolating the data using cubic or linear interpolation.
    :param T: The time vector.
    :param nuX: The nonuniform X vector.
    :param nuY: The nonuniform Y vector.
    :param Fs: The new uniform sampling rate.
    :param cubic: Whether to use cubic or linear interpolation.
    :return: A Tuple containing the new time vector, the new X vector, and the new Y vector.
    """
    # Cumulative sum of T minus the first element
    Tcumul = torch.cumsum(T, dim=0) - T[0]
    # Tmax is in microseconds
    Tmax = Tcumul[-1] 
    # Tmax is in seconds
    TmaxSeconds = Tmax / 1e6
    # Number of samples at Fs
    numSamples = int(TmaxSeconds * Fs)
    numSamples = numSamples if maxTimesteps is None else min(numSamples, maxTimesteps)
    # Create a time vector
    Tperiodic = torch.linspace(0, TmaxSeconds, numSamples)
    idxx = torch.searchsorted(Tcumul, Tperiodic * 1e6) - 1
    # idk why this is necessary but it is. TODO: investigate
    idxx[idxx < 0] = 0
    idxx[idxx > Tcumul.shape[0] - 2] = Tcumul.shape[0] - 2
    W = ((Tperiodic * 1e6 - Tcumul[idxx])/(Tcumul[idxx + 1] - Tcumul[idxx]))
    if resampleFn == "cubic":
        W0 = 2 * W ** 3 - 3 * W ** 2 + 1
        W1 = - 2 * W ** 3 + 3 * W ** 2
    if resampleFn == "linear":
        W0 = 1 - W
        W1 = W
    X = W0 * nuX[idxx] + W1 * nuX[idxx + 1]
    Y = W0 * nuY[idxx] + W1 * nuY[idxx + 1]
    return Tperiodic, X, Y

def project(X: torch.Tensor, Y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project the data onto a 2D plane using PCA.
    :param X: The X vector.
    :param Y: The Y vector.
    :return: A Tuple containing the projected X vector and the projected Y vector.
    """
    # Stack the X and Y vectors into a single data structure with 2 columns.
    stacked_data = torch.stack((X, Y), dim=1)

    # Create a PCA object and fit it to the data.
    pca = PCA(n_components=2)
    pca.fit(stacked_data)

    # Transform the data using the fitted PCA object.
    fitted_data = pca.transform(stacked_data)

    # Extract the first and second columns of the transformed data.
    projX = torch.from_numpy(fitted_data[:, 0])
    projY = torch.from_numpy(fitted_data[:, 1])

    # Return the projected X and Y vectors.
    return projX, projY

class STFTWrapper:
    """
    STFT class to hold STFT parameters, part of the preprocessing pipeline.
    :param NFFT: The number of FFT bins.
    :param HOP_LENGTH: The hop length between adjacent frames.
    :param WIN_LENGTH: The length of the STFT window.
    """
    def __init__(self, NFFT : int = 512, HOP_LENGTH : int = -1, WIN_LENGTH : int = -1):
        self.NFFT = NFFT
        if HOP_LENGTH == -1:
            self.HOP_LENGTH = NFFT // 2
        else:
            self.HOP_LENGTH = HOP_LENGTH
        if WIN_LENGTH == -1:
            self.WIN_LENGTH = NFFT
        else:
            self.WIN_LENGTH = WIN_LENGTH
    
    def __call__(self, X : torch.Tensor) -> torch.Tensor:
        return torch.stft(X, n_fft=self.NFFT, hop_length=self.HOP_LENGTH, win_length=self.WIN_LENGTH, center=False, pad_mode='reflect', normalized=False, onesided=True)

defaultSTFT = STFTWrapper(NFFT=512, HOP_LENGTH=256, WIN_LENGTH=512) # The default STFT object to use

def plotSpectrogram(X : torch.Tensor, Y : torch.Tensor, XOnly : bool = True, newFigure : bool = True, stftfunc : STFTWrapper = defaultSTFT):
    
    # Define a function to plot the spectrogram (we'll use it twice, this is just to make the code cleaner)
    def plotSpec(stft_sig):
        plt.imshow(torch.abs(stft_sig[:, :, 0]), origin='lower', aspect='auto')
    
    if newFigure:
        # Start figure    
        plt.figure(figsize=(10, 5))
    
    # If we're plotting both X and Y, make a 1x2 subplot
    if not XOnly:
        plt.subplot(1, 2, 1)
        
    # Plot X, regardless of whether we're plotting both X and Y
    stftX = stftfunc(X)
    plotSpec(stftX)
    
    # If we're plotting both X and Y, plot Y in the second subplot
    if not XOnly:
        stftY = stftfunc(Y)
        plt.subplot(1, 2, 2)    
        plotSpec(stftY)
    
    if newFigure:
        # Show the figure
        plt.show()

def processFromFile(fn, maxTimesteps = None, resampleFn = "cubic", skipPCA = False):
    """
    Convenience function to load, resample, and project data from a file. This is the typical use case of the preprocessing pipeline.
    """
    nuT, nuX, nuY = loadData(fn)
    T, X, Y = resample(nuT, nuX, nuY, maxTimesteps=maxTimesteps, resampleFn=resampleFn)
    if skipPCA:
        return T, X, Y
    else:
        projX, projY = project(X, Y)
        return T, projX, projY
