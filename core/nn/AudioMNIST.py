from glob import glob
import torch
import torch.nn.functional as F
import torchaudio
import os
import numpy as np
from .utils import trim_or_pad, trim_or_pad2
from ..signal.preprocess import processFromFile
from datasets import DatasetDict, Dataset

class AudioMnistDataset(torch.utils.data.Dataset):
    def __init__(self, wav_basedir='AudioMNIST/', csv_basedir='AudioMNIST/csv/', Fs=16000, device='cpu'):
        self.wav_basedir = wav_basedir
        self.csv_basedir = csv_basedir
        self.Fs = Fs
        self.wav_fns = glob(f"{wav_basedir}/data/*/*.wav")
        self.device = device

    def __len__(self):
        return len(self.wav_fns)
    
    def __getitem__(self, i):
        wav_fn = self.wav_fns[i]
        csv_fn = self._find_csv(wav_fn)
        if not csv_fn:
            raise FileNotFoundError(f"No corresponding CSV file found for {wav_fn}")


        wav, sr = torchaudio.load(wav_fn)
        wav = torchaudio.transforms.Resample(sr, self.Fs)(wav)
        wav = wav.squeeze().to(self.device)
     
        resp = processFromFile(csv_fn, Fs=self.Fs, method="sinc")
        if resp is not None:
            _, X, Y = resp
        else:
            X = torch.zeros_like(wav)
            Y = torch.zeros_like(wav)
        # Trim by 0.15 sec on both sides
        XY = torch.stack([X, Y], dim=0)
        XY = XY.to(self.device)
        XY = XY[:, int(self.Fs*0.15):int(-self.Fs*0.15)]
        # Pad if necessary
        if XY.shape[-1] < self.Fs:
            XY = F.pad(XY, (0, self.Fs - XY.shape[-1]))
        XY = XY.to(self.device)

        digit = int(os.path.basename((wav_fn.split("_")[-3][-1:])))
        speaker = int(os.path.basename(os.path.dirname(wav_fn)))

        return {"mouse_audio": XY, "original_audio": wav, "digit_target": digit, "speaker_target": speaker}
    
    def generator(self):
        for i in range(len(self)):
            yield self[i]
    
    def _find_csv(self, wav_fn):
        base_filename = os.path.basename(wav_fn).replace('.wav', '.csv')
        dirname = os.path.basename(os.path.dirname(wav_fn))
        potential_csv_path = os.path.join(self.csv_basedir, dirname, base_filename)
        if os.path.exists(potential_csv_path):
            return potential_csv_path
        print(potential_csv_path)
        return None

    def split_dataset(self, split=(0.85, 0.125, 0.025), seed=42):
        # Split into train, test, and dev
        train_split, test_split, dev_split = split
        assert train_split + test_split + dev_split == 1
        
        len_train = int(len(self) * train_split)
        len_test = int(len(self) * test_split)
        len_dev = len(self) - len_train - len_test
        previous_seed = np.random.seed()
        np.random.seed(seed)

        idx_train = np.random.choice(len(self), len_train, replace=False)
        idx_remaining = np.setdiff1d(np.arange(len(self)), idx_train)
        idx_test = np.random.choice(idx_remaining, len_test, replace=False)
        idx_dev = np.setdiff1d(idx_remaining, idx_test)
        assert len(idx_train) + len(idx_test) + len(idx_dev) == len(self)

        train = torch.utils.data.Subset(self, idx_train)
        test = torch.utils.data.Subset(self, idx_test)
        dev = torch.utils.data.Subset(self, idx_dev)
        
        np.random.seed(previous_seed)

        def generator_fn(ds):
            def gen():
                for i in range(len(ds)):
                    yield ds[i]
            return gen
        dataset = DatasetDict({
                "train": Dataset.from_generator(generator_fn(train)),
                "test": Dataset.from_generator(generator_fn(test)),
                "dev": Dataset.from_generator(generator_fn(dev))
            })

        return dataset

def get_audio_loaders(ds, batch_size=32, split=(0.85, 0.125, 0.025), device="cpu", Fs=16000, ret_labels=True):
    train, test, dev, _ = split_dataset(ds, batch_size=batch_size, split=split, device=device, Fs=Fs, ret_labels=ret_labels)

    def collate(batch):
        N = len(batch)
        out_wavs = torch.zeros((N, Fs), device=device)
        out_mouse = torch.zeros((N, 2, Fs), device=device)
        out_digit = []
        out_speaker = []

        for i, e in enumerate(batch):
            out_wavs[i] = trim_or_pad(e[1].squeeze(), Fs).to(device)
            out_mouse[i] = trim_or_pad2(e[0].squeeze(), Fs).to(device)
            out_digit.append(e[2])
            out_speaker.append(e[3])

        out_digit = torch.tensor(out_digit, device=device)
        out_speaker = torch.tensor(out_speaker, device=device)

        if ret_labels:
            return out_mouse, out_wavs, out_speaker, out_digit
        return out_mouse, out_wavs

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    dev_loader = torch.utils.data.DataLoader(dev, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0)

    print(f"Train: {len(train_loader)} batches = {len(train_loader) * batch_size} samples")
    print(f"Test: {len(test_loader)} batches = {len(test_loader) * batch_size} samples")
    print(f"Dev: {len(dev_loader)} batches = {len(dev_loader) * batch_size} samples")

    return train_loader, test_loader, dev_loader
