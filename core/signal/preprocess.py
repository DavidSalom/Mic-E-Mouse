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

def resample(nuT : torch.Tensor, nuX : torch.Tensor, nuY : torch.Tensor, Fs : int = 16000, method : str = "cubic", device : str = "cpu") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Resample the signal to uniformize the sampling rate.
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
    # Number of samples at Fs
    numSamples = int(Tmax * Fs / 1e6)
    # Create a uniform-time vector
    Tperiodic = torch.linspace(0, Tmax, numSamples, device = device)

    idxx = torch.searchsorted(Tcumul, Tperiodic) - 1
    # idk why this is necessary but it is. TODO: investigate
    # idxx[idxx < 0] = 0
    # idxx[idxx > Tcumul.shape[0] - 2] = Tcumul.shape[0] - 2
    X = torch.zeros_like(Tperiodic, device = device)
    Y = torch.zeros_like(Tperiodic, device = device)
    if method == "nearest":
        X = nuX[idxx]
        Y = nuY[idxx]
    if method == "linear":
        W = ((Tperiodic - Tcumul[idxx])/(Tcumul[idxx + 1] - Tcumul[idxx]))
        W0 = 1 - W
        W1 = W
        X = W0 * nuX[idxx] + W1 * nuX[idxx + 1]
        Y = W0 * nuY[idxx] + W1 * nuY[idxx + 1]
    if method == "cubic":
        W = ((Tperiodic - Tcumul[idxx])/(Tcumul[idxx + 1] - Tcumul[idxx]))
        W0 = 2 * W ** 3 - 3 * W ** 2 + 1
        W1 = - 2 * W ** 3 + 3 * W ** 2
        X = W0 * nuX[idxx] + W1 * nuX[idxx + 1]
        Y = W0 * nuY[idxx] + W1 * nuY[idxx + 1]
    if method == "sinc":
        W = ((Tperiodic - Tcumul[idxx])/(Tcumul[idxx + 1] - Tcumul[idxx]))
        W0 = sinc((1 - W) * np.pi)
        W1 = sinc(W * np.pi)
        X = W0 * nuX[idxx] + W1 * nuX[idxx + 1]
        Y = W0 * nuY[idxx] + W1 * nuY[idxx + 1]
    return Tperiodic, X, Y

def processFromFile(fn, Fs = 16000, method = "cubic", device = 'cpu'):
    """
    Convenience function to load, and resample data from a file.
    """
    nuT, nuX, nuY = loadData(fn)
    nuT = nuT.to(device)
    nuX = nuX.to(device)
    nuY = nuY.to(device)
    T, X, Y = resample(nuT, nuX, nuY, Fs=Fs, method=method, device = device)
    return T, X, Y