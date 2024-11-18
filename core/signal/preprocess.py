import numpy as np
import torch
from typing import Tuple, Optional

def load_data(src: str) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Load the data from the source file and return it as PyTorch tensors.

    :param src: The path to the source file.
    :return: A tuple containing the time vector, the nonuniform X vector, and the nonuniform Y vector,
             or None if an error occurs.
    """
    try:
        # Load the data from the source file.
        # Assuming the data is floating-point numbers with a header row.
        data = np.loadtxt(src, delimiter=',', skiprows=1, dtype=float)
        # Convert the data into PyTorch tensors.
        data = torch.from_numpy(data)
        # Extract the time and nonuniform X, Y vectors.
        nuT = data[:, 0]
        nuX = data[:, 1]
        nuY = data[:, 2]
    except OSError as e:
        # Handle file not found or inaccessible.
        print(f"File error: {e}")
        return None
    except IndexError as e:
        # Handle incorrect data format.
        print(f"Data format error: {e}")
        return None
    except Exception as e:
        # Catch-all for any other exceptions.
        print(f"Unexpected error: {e}")
        return None

    return nuT, nuX, nuY

def sinc(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the sinc function, handling the case where x is zero.

    :param x: Input tensor.
    :return: The sinc of x.
    """
    return torch.where(x == 0, torch.ones_like(x), torch.sin(x) / x)

def resample(
    nuT: torch.Tensor,
    nuX: torch.Tensor,
    nuY: torch.Tensor,
    Fs: int = 16000,
    method: str = "cubic",
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Resample the signal to a uniform sampling rate.

    :param nuT: The nonuniform time vector.
    :param nuX: The nonuniform X vector.
    :param nuY: The nonuniform Y vector.
    :param Fs: The new uniform sampling rate in Hz.
    :param method: Interpolation method ('nearest', 'linear', 'cubic', 'sinc').
    :param device: The device to perform computations on ('cpu' or 'cuda').
    :return: A tuple containing the new time vector, the resampled X vector, and the resampled Y vector.
    """
    # Move tensors to the specified device.
    nuT = nuT.to(device)
    nuX = nuX.to(device)
    nuY = nuY.to(device)

    # Compute cumulative time vector starting from zero.
    Tcumul = torch.cumsum(nuT, dim=0) - nuT[0]
    # Total duration in microseconds.
    Tmax = Tcumul[-1]
    # Compute the number of samples based on the desired sampling frequency.
    num_samples = int(Tmax * Fs / 1e6)
    # Create a uniform time vector over the duration.
    T_uniform = torch.linspace(0, Tmax, num_samples, device=device)

    # Find the indices in Tcumul that correspond to the uniform time steps.
    idxx = torch.searchsorted(Tcumul, T_uniform, right=True) - 1
    # Ensure indices are within valid range.
    idxx = idxx.clamp(min=0, max=Tcumul.shape[0] - 2)

    # Initialize output tensors.
    X_resampled = torch.zeros_like(T_uniform, device=device)
    Y_resampled = torch.zeros_like(T_uniform, device=device)

    # Compute interpolation weights.
    delta_T = Tcumul[idxx + 1] - Tcumul[idxx]
    W = (T_uniform - Tcumul[idxx]) / delta_T

    if method == "nearest":
        # Nearest neighbor interpolation.
        X_resampled = nuX[idxx]
        Y_resampled = nuY[idxx]
    elif method == "linear":
        # Linear interpolation.
        X_resampled = (1 - W) * nuX[idxx] + W * nuX[idxx + 1]
        Y_resampled = (1 - W) * nuY[idxx] + W * nuY[idxx + 1]
    elif method == "cubic":
        # Cubic interpolation coefficients.
        W0 = 2 * W ** 3 - 3 * W ** 2 + 1
        W1 = -2 * W ** 3 + 3 * W ** 2
        X_resampled = W0 * nuX[idxx] + W1 * nuX[idxx + 1]
        Y_resampled = W0 * nuY[idxx] + W1 * nuY[idxx + 1]
    elif method == "sinc":
        # Sinc interpolation.
        W0 = sinc((1 - W) * torch.pi)
        W1 = sinc(W * torch.pi)
        X_resampled = W0 * nuX[idxx] + W1 * nuX[idxx + 1]
        Y_resampled = W0 * nuY[idxx] + W1 * nuY[idxx + 1]
    else:
        # Handle unknown interpolation methods.
        raise ValueError(f"Unknown interpolation method: {method}")

    return T_uniform, X_resampled, Y_resampled

def process_from_file(
    filename: str,
    Fs: int = 16000,
    method: str = "cubic",
    device: str = 'cpu'
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Load data from a file and resample it.

    :param filename: The path to the data file.
    :param Fs: The new uniform sampling rate in Hz.
    :param method: Interpolation method ('nearest', 'linear', 'cubic', 'sinc').
    :param device: The device to perform computations on ('cpu' or 'cuda').
    :return: A tuple containing the new time vector, the resampled X vector, and the resampled Y vector,
             or None if an error occurs.
    """
    # Load data from the file.
    data = load_data(filename)
    if data is not None:
        nuT, nuX, nuY = data
        # Resample the data.
        return resample(nuT, nuX, nuY, Fs=Fs, method=method, device=device)
    else:
        # Return None if loading data failed.
        return None
