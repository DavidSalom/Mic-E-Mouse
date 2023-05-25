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
        """
        VCTK_root_path: path to the VCTK dataset parent folder, which should contain wav48 and txt folders
        """
        self.VCTK_root_path = VCTK_root_path
        
        # Initialize the dataset
        self._init_ds()
        
        if 'pad' in kwargs:
            assert isinstance(kwargs['pad'], bool), "pad must be a boolean"
            self.pad = kwargs['pad']
        else:
            self.pad = True
        
        if 'length_seconds' in kwargs:
            assert isinstance(kwargs['length_seconds'], int), "length_seconds must be an integer"
            self.length_seconds = kwargs['length_seconds']
        else:
            self.length_seconds = 5
        
        if 'normalize' in kwargs:
            assert isinstance(kwargs['normalize'], bool), "normalize must be a boolean"
            self.normalize = kwargs['normalize']
        else:
            self.normalize = True
            
        if "resample" in kwargs:
            assert isinstance(kwargs['resample'], bool), "resample must be a boolean"
            self.resample = kwargs['resample']
        else:
            self.resample = True
        
        if "resample_rate" in kwargs:
            assert isinstance(kwargs['resample_rate'], int), "resample_rate must be an integer"
            self.resample_rate = kwargs['resample_rate']
        else:
            self.resample_rate = 16000
            
    def _normalize(self, waveform):
        waveform = waveform - waveform.mean()
        power = waveform.pow(2).mean()
        waveform = waveform / power.sqrt()
        return waveform
        
    def _init_ds(self):
        # Get all the wav files paths
        self.wav_files_paths = glob.glob(os.path.join(self.VCTK_root_path, 'wav48', '*', '*.wav'))
        self.txt_files_paths = [wav_file_path.replace('wav48', 'txt').replace('.wav', '.txt') for wav_file_path in self.wav_files_paths]
        
        # Remove the files that don't have a corresponding txt file
        self.wav_files_paths = [wav_file_path for wav_file_path, txt_file_path in zip(self.wav_files_paths, self.txt_files_paths) if os.path.exists(txt_file_path)]
        self.txt_files_paths = [wav_file_path.replace('wav48', 'txt').replace('.wav', '.txt') for wav_file_path in self.wav_files_paths]
        
        # Get the speaker IDs
        self.speaker_ids = [os.path.basename(os.path.dirname(wav_file)) for wav_file in self.wav_files_paths]
        
        # Get the utterance IDs
        self.utterance_ids = [os.path.basename(wav_file).split('.')[0] for wav_file in self.wav_files_paths]
        
        # Load speaker info
        self.speaker_info = self._process_info()
    
    def _process_info(self):
        """
        Load the speaker info from the speaker-info.txt file. Used in init.
        """
        speaker_info = {}
        with open(os.path.join(self.VCTK_root_path, 'speaker-info.txt'), 'r') as f:
            lines = f.readlines()
            lines = [line.strip().split() for line in lines]
            lines = lines[1:]
            for line in lines:
                id = line[0]
                age= int(line[1])
                gender= line[2]
                accent= line[3]
                regions= " ".join(line[4:])
                speaker_info[id] = {
                    'age': age,
                    'gender': gender,
                    'accent': accent,
                    'regions': regions
                }
        return speaker_info
    
    def _load_audio(self, wav_file_path : str):
        """
        Load the audio file from the wav file path. Used in __getitem__ and _init_ds.
        wav_file_path: str
        """
        waveform, sample_rate = torchaudio.load(wav_file_path)
        return waveform, sample_rate
    
    def _load_text(self, txt_file_path : str):
        """
        Load the text file from the txt file path. Used in __getitem__.
        """
        with open(txt_file_path, 'r') as f:
            text = f.read().strip()
        return text
    
    def _load_metadata(self, speaker_id):
        try:
            data = self.speaker_info[speaker_id]
        except KeyError:
            data = {
                    'age': -1,
                    'gender': "X",
                    'accent': "X",
                    'regions': "X"
                }
        return data
    
    def __len__(self):
        return len(self.idx_to_wav_offset_dict)

    def __getitem__(self, idx):
        wav_file_path = self.wav_files_paths[idx] # wav file path
        txt_file_path = self.txt_files_paths[idx] # txt file path

        speaker_id = self.speaker_ids[idx].replace('p', '') # speaker id, making sure to remove the 'p' at the beginning

        # utterance_id = self.utterance_ids[idx]
        
        waveform, sample_rate = self._load_audio(wav_file_path)
        
        if self.resample:
            waveform, sample_rate = self._resample(waveform, sample_rate, self.resample_rate)
        
        text = self._load_text(txt_file_path)
        metadata = self._load_metadata(speaker_id)
        
        if self.normalize:
            waveform = self._normalize(waveform)
        
        if self.pad:
            waveform = self._pad(waveform, sample_rate, self.length_seconds)
        
        return waveform, sample_rate, text, metadata
    
    def get_speaker_info(self, speaker_id):
        return self._load_metadata(speaker_id)

    def _resample(self, waveform: torch.Tensor, sample_rate : int, new_samplerate: int):
        resampled_waveform = torchaudio.transforms.Resample(sample_rate, new_samplerate)(waveform)
        return resampled_waveform, new_samplerate

    def _pad(self, waveform : torch.Tensor, sample_rate : int, length_seconds:int=10):
        """
        Pad or clip the waveform to the desired length in seconds, and add an indicator in the second channel to mark which part is the original waveform.
        waveform: torch.Tensor, shape=(1, n_samples)
        sample_rate: int
        length_seconds: int 
        """
        padded_waveform = torch.nn.functional.pad(waveform, (0, int(length_seconds*sample_rate)))
        if padded_waveform.shape[1] > int(length_seconds*sample_rate):
            padded_waveform = padded_waveform[:, :int(length_seconds*sample_rate)]
            padded_waveform = self._normalize(padded_waveform)        
        padded_indicator = torch.zeros_like(padded_waveform)
        padded_indicator[:, :waveform.shape[1]] = 1
        padded_indicator[:, waveform.shape[1]:] = 0
        fused_waveform = torch.cat([padded_waveform, padded_indicator], dim=0)
        return fused_waveform