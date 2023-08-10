"""
VCTK_dataloader
=====
Provides a PyTorch Dataset for loading and preprocessing the VCTK speech dataset.
"""
import torch
import pandas as pd
import os
import glob
import torchaudio
import numpy as np
from math import ceil
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from .utils import CacheMixin
import logging

class VCTK_Dataset(Dataset, CacheMixin):
    """
    A PyTorch Dataset for loading and preprocessing the VCTK speech dataset.
    
    Attributes:
        VCTK_root_path (str): Path to the VCTK dataset parent folder, which should contain wav48 and txt folders.
        resample_rate (int, optional): The rate to resample the audio files to. Defaults to 8000.
        wav_files_paths (list of str): The paths to all wav files in the VCTK dataset.
        idx_to_wav_offset_dict (dict): A dictionary mapping from sample index to a tuple of wav file index and offset.
    """

    def __init__(self, VCTK_root_path : str, **kwargs : dict[str, any]):
        """
        Args:
            VCTK_root_path (str): Path to the VCTK dataset parent folder, which should contain wav48 and txt folders.
            **kwargs: Additional keyword arguments. Currently, only "resample_rate" is used.
        """
        super(VCTK_Dataset, self).__init__()
        logging.info("Initializing VCTK_Dataset...")
        self.VCTK_root_path = VCTK_root_path
        self._init_ds()
        
        self.clip_min = -10.4615
        self.clip_max = 11.3003
        
        self.resample_rate = kwargs.get('resample_rate', 8000)

    def _init_ds(self, other) -> None:
        self.VCTK_root_path = other.VCTK_root_path
        self.wav_files_paths = other.wav_files_paths
        self.idx_to_wav_offset_dict = other.idx_to_wav_offset_dict
        
    def _init_ds(self) -> None:
        """
        Initialize the dataset by loading all wav file paths and building the index-to-wav-offset dictionary.
        """
        self.wav_files_paths = glob.glob(os.path.join(self.VCTK_root_path, 'wav48', '*', '*.wav'))
        self.idx_to_wav_offset_dict = {}
        self._build_idx_to_wav_dict()

    def _build_idx_to_wav_dict(self)-> None:
        """
        Build a dictionary mapping from sample index to a tuple of wav file index and offset.
        """
        logging.info("Building the idx to wav offset dict...")
        largest_idx = 0
        pbar = tqdm(total=len(self.wav_files_paths))
        for wav_idx, wav_fn in enumerate(self.wav_files_paths):
            wav, sr = self._load_audio(wav_fn)
            number_of_one_second_chunks = wav.shape[1] // sr
            if wav.shape[1] % sr >= 0.8 * sr:
                number_of_one_second_chunks += 1
            for i in range(number_of_one_second_chunks):
                self.idx_to_wav_offset_dict[largest_idx] = (wav_idx, i)
                largest_idx += 1
            pbar.update(1)
        pbar.close()
    
    def _load_audio(self, wav_file_path : str) -> tuple[torch.FloatTensor, int]:
        """
        Load an audio file from the given path.

        Args:
            wav_file_path (str): The path to the wav file to load.

        Returns:
            tuple: A tuple containing the waveform (torch.FloatTensor) and the sample rate (int).
        """
        waveform, sample_rate = torchaudio.load(wav_file_path)
        return waveform, sample_rate
    
    def _normalize(self, waveform : torch.FloatTensor) -> torch.FloatTensor:
        """
        Normalize a waveform by subtracting the mean and dividing by the standard deviation.

        Args:
            waveform (torch.FloatTensor): The waveform to normalize.

        Returns:
            torch.FloatTensor: The normalized waveform.
        """
        waveform = waveform - waveform.mean()
        power = waveform.pow(2).mean()
        waveform = waveform / power.sqrt()
        return waveform
    
    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.idx_to_wav_offset_dict)

    def __getitem__(self, idx : int) -> tuple[torch.FloatTensor, int]:
        """
        Get a sample from the dataset.

        Args:
            idx (int): The index of the sample to get.

        Returns:
            tuple: A tuple containing the waveform (torch.FloatTensor) and the sample rate (int).
        """
        idx = list(self.idx_to_wav_offset_dict.keys())[idx]
        wav_idx, wav_offset = self.idx_to_wav_offset_dict[idx]
        wav_file_path = self.wav_files_paths[wav_idx] # wav file path
        
        waveform, sample_rate = self._load_audio(wav_file_path)
        waveform = self._normalize(waveform)
        waveform = waveform[:, wav_offset*sample_rate:(wav_offset+1)*sample_rate]
        waveform, sample_rate = self._resample(waveform, sample_rate, self.resample_rate)

        if waveform.shape[1] < sample_rate:
            waveform = torch.nn.functional.pad(waveform, (0, sample_rate - waveform.shape[1]))    
        
        # Normalize to zero mean and unit variance
        waveform = (waveform - waveform.mean())/waveform.std()
        # Clip to computed min and max
        waveform = torch.clip(waveform, self.clip_min, self.clip_max)
        # Normalize to [-1, 1]
        waveform = 2 * (waveform - self.clip_min) / (self.clip_max - self.clip_min) - 1
        
        return waveform, sample_rate
    
    def _resample(self, waveform: torch.FloatTensor, sample_rate : int, new_samplerate: int) -> tuple[torch.FloatTensor, int]:
        """
        Resample a waveform to a new sample rate.

        Args:
            waveform (torch.FloatTensor): The waveform to resample.
            sample_rate (int): The current sample rate of the waveform.
            new_samplerate (int): The new sample rate to resample to.

        Returns:
            tuple: A tuple containing the resampled waveform (torch.Tensor) and the new sample rate (int).
        """    
        resampled_waveform = torchaudio.transforms.Resample(sample_rate, new_samplerate)(waveform)
        return resampled_waveform, new_samplerate