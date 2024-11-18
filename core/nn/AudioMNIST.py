import os
from glob import glob
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, Subset
from .utils import trim_or_pad, trim_or_pad2
from ..signal.preprocess import process_from_file
from datasets import DatasetDict, Dataset

class AudioMnistDataset(Dataset):
    def __init__(self, wav_basedir='AudioMNIST/', csv_basedir='AudioMNIST/csv/', Fs=16000, device = 'cpu'):
        self.wav_basedir = wav_basedir
        self.csv_basedir = csv_basedir
        self.Fs = Fs
        self.wav_fns = glob(os.path.join(wav_basedir, 'data', '*', '*.wav'))
        self.device = device

    def __len__(self):
        return len(self.wav_fns)
    
    def __getitem__(self, idx):
        wav_fn = self.wav_fns[idx]
        csv_fn = self._find_csv(wav_fn)
        if not csv_fn:
            raise FileNotFoundError(f"No corresponding CSV file found for {wav_fn}")

        # Load and process the audio waveform
        wav, sr = torchaudio.load(wav_fn)
        wav = torchaudio.transforms.Resample(sr, self.Fs)(wav).squeeze().to(self.device)

        # Process the response from the CSV file
        resp = process_from_file(csv_fn, Fs=self.Fs, method="sinc")
        if resp is None:
            raise ValueError(f"Failed to process response from {csv_fn}")

        # Unpack the response
        T_uniform, X_resampled, Y_resampled = resp

        # Ensure that the lengths of the resampled signals and the audio waveform match
        min_length = min(len(wav), len(X_resampled))

        # Trim or pad the signals to have the same length
        wav = wav[:min_length]
        X_resampled = X_resampled[:min_length]
        Y_resampled = Y_resampled[:min_length]

        # Stack the resampled X and Y signals
        XY = torch.stack([X_resampled, Y_resampled], dim=0)

        # Extract digit and speaker information from filename
        filename_parts = os.path.basename(wav_fn).split('_')
        digit = int(filename_parts[-3][-1])
        speaker = int(os.path.basename(os.path.dirname(wav_fn)))

        return {
            "mouse_audio": XY,
            "original_audio": wav,
            "digit_target": digit,
            "speaker_target": speaker
        }

    
    def _find_csv(self, wav_fn):
        base_filename = os.path.basename(wav_fn).replace('.wav', '.csv')
        dirname = os.path.basename(os.path.dirname(wav_fn))
        potential_csv_path = os.path.join(self.csv_basedir, dirname, base_filename)
        if os.path.exists(potential_csv_path):
            return potential_csv_path
        else:
            return None
    
    def split_dataset(self, split_ratios=(0.85, 0.125, 0.025), seed=42):
        assert sum(split_ratios) == 1.0, "Split ratios must sum to 1.0"
        previous_seed = np.random.seed()
        np.random.seed(seed)
        dataset_size = len(self)
        indices = np.random.permutation(dataset_size)
        train_end = int(split_ratios[0] * dataset_size)
        test_end = train_end + int(split_ratios[1] * dataset_size)
        
        train_indices = indices[:train_end]
        test_indices = indices[train_end:test_end]
        dev_indices = indices[test_end:]
        
        train_dataset = Subset(self, train_indices)
        test_dataset = Subset(self, test_indices)
        dev_dataset = Subset(self, dev_indices)
        np.random.seed(previous_seed)

        def generator_fn(ds):
            def gen():
                for i in range(len(ds)):
                    yield ds[i]
            return gen
        dataset = DatasetDict({
                "train": Dataset.from_generator(generator_fn(train_dataset)),
                "test": Dataset.from_generator(generator_fn(test_dataset)),
                "dev": Dataset.from_generator(generator_fn(dev_dataset))
            })

        return dataset
        return train_dataset, test_dataset, dev_dataset

def collate_fn(batch):
    # Extract batch elements
    mouse_audios = [item['mouse_audio'] for item in batch]
    original_audios = [item['original_audio'] for item in batch]
    digit_targets = [item['digit_target'] for item in batch]
    speaker_targets = [item['speaker_target'] for item in batch]

    # Pad or trim audio tensors
    Fs = 16000  # Assuming sample rate is 16kHz
    max_length = Fs  # Define the desired length in samples

    # Pad or trim mouse_audios
    mouse_audios = [trim_or_pad2(audio, max_length) for audio in mouse_audios]
    mouse_audios = torch.stack(mouse_audios)

    # Pad or trim original_audios
    original_audios = [trim_or_pad(audio, max_length) for audio in original_audios]
    original_audios = torch.stack(original_audios)

    # Stack targets
    digit_targets = torch.tensor(digit_targets, dtype=torch.long)
    speaker_targets = torch.tensor(speaker_targets, dtype=torch.long)

    return mouse_audios, original_audios, speaker_targets, digit_targets


def get_audio_loaders(dataset, batch_size=32, split_ratios=(0.85, 0.125, 0.025), num_workers=0):
    # Split the dataset
    train_dataset, test_dataset, dev_dataset = dataset.split_dataset(split_ratios)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)

    print(f"Train: {len(train_loader.dataset)} samples")
    print(f"Test: {len(test_loader.dataset)} samples")
    print(f"Dev: {len(dev_loader.dataset)} samples")

    return train_loader, test_loader, dev_loader
