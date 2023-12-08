import numpy as np
import matplotlib.pyplot as plt
import torch
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


def sinc(x):
    """Helper function to compute the sinc function."""
    return torch.where(x == 0, torch.ones_like(x), torch.sin(x) / x)

def resample(nuT : torch.Tensor, nuX : torch.Tensor, nuY : torch.Tensor, Fs : int = 16000) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Resample the signal to uniformize the sampling rate. Implements https://dl.acm.org/doi/10.1109/78.869037 for sinc-based resampling.
    :param T: The nonuniform time vector.
    :param nuX: The nonuniform X vector.
    :param nuY: The nonuniform Y vector.
    :param Fs: The new uniform sampling rate.
    :return: A Tuple containing the new time vector, the new X vector, and the new Y vector.
    """
    # Cumulative sum of T, re-indexed to start at 0
    Tcumul = torch.cumsum(nuT, dim=0) - nuT[0]
    # Tmax is in microseconds, the offset for the largest time in the signal
    Tmax = Tcumul[-1] 
    # Tmax is in seconds
    TmaxSeconds = Tmax / 1e6
    # Number of samples at Fs
    numSamples = int(TmaxSeconds * Fs)
    # Create a uniform-time vector
    Tperiodic = torch.linspace(0, TmaxSeconds, numSamples)
    idxx = torch.searchsorted(Tcumul, Tperiodic * 1e6) - 1

    X = torch.zeros_like(Tperiodic)
    Y = torch.zeros_like(Tperiodic)
    for i, t in tqdm(enumerate(Tperiodic)):
        sinc_coeffs = sinc(torch.pi * Fs * (t - Tcumul / 1e6))
        X[i] = torch.sum(nuX * sinc_coeffs)
        Y[i] = torch.sum(nuY * sinc_coeffs)
    
    return Tperiodic, X, Y

def processFromFile(fn, Fs = 16000):
    """
    Convenience function to load, and resample data from a file.
    """
    nuT, nuX, nuY = loadData(fn)
    T, X, Y = resample(nuT, nuX, nuY, Fs=Fs)
    return T, X, Y