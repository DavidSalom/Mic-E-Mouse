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
        self.melFn = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=80,
            f_min=20,
            f_max=4000,
            # power=2.0,
            normalized=True,
        )
        self.melFn = self.melFn.to(device)
        self.device = device
    def __len__(self):
        return len(self.wav_fns)
    
    def __getitem__(self, i):
        wav_fn = self.wav_fns[i]
        csv_fn = wav_fn.replace('.wav', '.csv').replace('data', 'gen/csv')
        wav, sr = torchaudio.load(wav_fn)
        wav = torchaudio.transforms.Resample(sr, self.Fs)(wav)
        wav = wav.squeeze().to(self.device)
        _, X, Y = processFromFile(csv_fn, Fs = self.Fs, method="cubic")
        # trim by 1 sec on both sides
        XY = torch.stack([X, Y], axis=1)
        XY = XY[self.Fs:-self.Fs]
        # XY is (T, 2), sum to get (T,)
        XY = XY.sum(-1) / 2
        XY = XY.to(self.device)
        XYmel = self.melFn(XY)
        # pad last dim to 120
        # XYmel = F.pad(XYmel, (0, 120 - XYmel.shape[-1]))

        wavMel = self.melFn(wav)
        # wavMel = F.pad(wavMel, (0, 120 - wavMel.shape[-1]))

        digit = int(os.path.basename((wav_fn.split("_")[0])))
        speaker = int(os.path.basename(os.path.dirname(wav_fn)))
        
        # normalize XYmel and wavMel
        # XYmel = (XYmel - XYmel.mean()) / XYmel.std()
        # wavMel = (wavMel - wavMel.mean()) / wavMel.std()

        return XYmel, wavMel, digit, speaker