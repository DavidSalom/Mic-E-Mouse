import torch
import pandas as pd
import os
import glob
import torchaudio
import numpy as np
from math import ceil
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class VCTK_Dataset(Dataset):
    def __init__(self, VCTK_root_path : str, **kwargs):
        super(VCTK_Dataset, self).__init__()
        """
        VCTK_root_path: path to the VCTK dataset parent folder, which should contain wav48 and txt folders
        """
        self.VCTK_root_path = VCTK_root_path
        
        # Initialize the dataset
        self._init_ds()
        
        self.resample_rate = kwargs.get('resample_rate', 8000)
            
    def _init_ds(self):
        # Get all the wav files paths
        self.wav_files_paths = glob.glob(os.path.join(self.VCTK_root_path, 'wav48', '*', '*.wav'))
        
        self.idx_to_wav_offset_dict = {}
        self._build_idx_to_wav_dict()
    
    def _build_idx_to_wav_dict(self):
        largest_idx = 0
        print("Building the idx to wav offset dict...")
        pbar = tqdm(total=len(self.wav_files_paths))
        for wav_idx, wav_fn in enumerate(self.wav_files_paths):
            wav, sr = self._load_audio(wav_fn)
            number_of_one_second_chunks = wav.shape[1] // sr
            # Check the last chunk if it is at least 0.8 seconds long
            if wav.shape[1] % sr >= 0.8 * sr:
                number_of_one_second_chunks += 1
            for i in range(number_of_one_second_chunks):
                self.idx_to_wav_offset_dict[largest_idx] = (wav_idx, i)
                largest_idx += 1
            pbar.update(1)
        pbar.close()
    
    def _load_audio(self, wav_file_path : str):
        """
        Load the audio file from the wav file path. Used in __getitem__ and _init_ds.
        wav_file_path: str
        """
        waveform, sample_rate = torchaudio.load(wav_file_path)
        return waveform, sample_rate
    
    def _normalize(self, waveform):
        waveform = waveform - waveform.mean()
        power = waveform.pow(2).mean()
        waveform = waveform / power.sqrt()
        return waveform
    
    def __len__(self):
        return len(self.idx_to_wav_offset_dict)

    def __getitem__(self, idx):
        idx = list(self.idx_to_wav_offset_dict.keys())[idx]
        wav_idx, wav_offset = self.idx_to_wav_offset_dict[idx]
        wav_file_path = self.wav_files_paths[wav_idx] # wav file path
        
        waveform, sample_rate = self._load_audio(wav_file_path)
        waveform = self._normalize(waveform)
        waveform = waveform[:, wav_offset*sample_rate:(wav_offset+1)*sample_rate]
        waveform, sample_rate = self._resample(waveform, sample_rate, self.resample_rate)

        if waveform.shape[1] < sample_rate:
            waveform = torch.nn.functional.pad(waveform, (0, sample_rate - waveform.shape[1]))    
        
        return waveform, sample_rate
    
    def _resample(self, waveform: torch.Tensor, sample_rate : int, new_samplerate: int):
        resampled_waveform = torchaudio.transforms.Resample(sample_rate, new_samplerate)(waveform)
        return resampled_waveform, new_samplerate