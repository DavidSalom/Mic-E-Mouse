from glob import glob
import torch
import torch.nn.functional as F
import torchaudio
from ..signal.preprocess import processFromFile
import os


class audioMnistDataset(torch.utils.data.Dataset):
    def __init__(self, basedir = 'AudioMNIST/', Fs = 16000, device = 'cpu'):
        self.basedir = basedir
        self.Fs = Fs
        self.wav_fns = glob(f"{basedir}/data/*/*.wav")
        self.device = device
    def __len__(self):
        return len(self.wav_fns)
    
    def __getitem__(self, i):
        wav_fn = self.wav_fns[i]
        csv_fn = wav_fn.replace('.wav', '.csv').replace('data', 'gen/csv')
        wav, sr = torchaudio.load(wav_fn)
        wav = torchaudio.transforms.Resample(sr, self.Fs)(wav)
        wav = wav.squeeze().to(self.device)
        _, X, Y = processFromFile(csv_fn, Fs = self.Fs, method="sinc")
        # trim by 1 sec on both sides
        XY = torch.stack([X, Y], axis=1)
        XY = XY[self.Fs:-self.Fs]
        # XY is (T, 2), sum to get (T,)
        XY = XY.to(self.device)

        digit = int(os.path.basename((wav_fn.split("_")[0])))
        speaker = int(os.path.basename(os.path.dirname(wav_fn)))

        return XY, wav, digit, speaker