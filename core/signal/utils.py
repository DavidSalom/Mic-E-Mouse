import torch
def DCT(signal):
    """
    Perform the Discrete Cosine Transform (DCT) using FFT in PyTorch.
    """
    # Extend by flipping signal
    extended_signal = torch.cat([signal, signal.flip(0)])
    # Apply FFT
    fft_result = torch.fft.fft(extended_signal)
    # The real part corresponds to the DCT
    dct_result = fft_result.real
    # Only the first half of the result is needed (because of symmetry)
    return dct_result[:len(signal)]

def DCT2D(image):
    """
    Perform the 2D Discrete Cosine Transform (DCT) using FFT in PyTorch.
    """
    # Apply DCT via FFT along rows
    dct_rows = torch.stack([DCT(row) for row in image])

    # Apply DCT via FFT along columns
    dct_cols = torch.stack([DCT(col) for col in dct_rows.t()]).t()
    return dct_cols