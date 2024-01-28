from glob import glob
import torch
import torch.nn.functional as F
import torchaudio
from ..signal.preprocess import processFromFile
import os
import numpy as np
from .utils import trim_or_pad, trim_or_pad2


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
        XY = torch.stack([X, Y], dim = 0)
        XY = XY[:, self.Fs:-self.Fs]
        XY = XY.to(self.device)

        digit = int(os.path.basename((wav_fn.split("_")[0])))
        speaker = int(os.path.basename(os.path.dirname(wav_fn)))

        return XY, wav, digit, speaker
    
def getAudioLoaders(ds, batch_size = 32, split = (0.85, 0.125, 0.025), device = "cpu", Fs = 16000, ret_labels = True):
    # Split into train, test, and dev
    trainSplit, testSplit, devSplit = split
    assert trainSplit + testSplit + devSplit == 1
    
    lenTrain = int(len(ds) * trainSplit)
    lenTest = int(len(ds) * testSplit)
    lenDev = len(ds) - lenTrain - lenTest

    idxtrain = np.random.choice(len(ds), lenTrain, replace=False)
    idxTest = np.setdiff1d(np.arange(len(ds)), idxtrain)
    idxtest = np.random.choice(idxTest, lenTest, replace=False)
    idxdev = np.setdiff1d(idxTest, idxtest)
    assert len(idxtrain) + len(idxtest) + len(idxdev) == len(ds)

    train = torch.utils.data.Subset(ds, idxtrain)
    test = torch.utils.data.Subset(ds, idxtest)
    dev = torch.utils.data.Subset(ds, idxdev)

    # T = 5 * Fs
    T = 501
    n_mels = 80
    def collate(batch):
        N = len(batch)
        out_wavs = torch.zeros((N, 1 * Fs), device=device)
        out_mouse = torch.zeros((N, 2, 1 * Fs), device=device)
        out_digit = []
        out_speaker = []
        i = 0
        for e in batch:
            out_wavs[i] = trim_or_pad(e[1].squeeze(), 1 * Fs).to(device)
            out_mouse[i] = trim_or_pad2(e[0].squeeze(), 1 * Fs).to(device)

            out_digit.append(e[2])
            out_speaker.append(e[3])

            i += 1
        if ret_labels:
            return out_mouse, out_wavs, out_digit, out_speaker
        return out_mouse, out_wavs


    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    dev_loader = torch.utils.data.DataLoader(dev, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    print(f"Train: {len(train_loader)} batches = {len(train_loader) * batch_size} samples")
    print(f"Test: {len(test_loader)} batches = {len(test_loader) * batch_size} samples")
    print(f"Dev: {len(dev_loader)} batches = {len(dev_loader) * batch_size} samples")

    return train_loader, test_loader, dev_loader