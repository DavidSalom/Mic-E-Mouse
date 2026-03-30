import os
from typing import Tuple

import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
from ..signal.preprocess import *
import logging
from .utils import trim_or_pad, trim_or_pad2

SampleType = Tuple[Tensor, int, str, str, str]


class VCTK_CSV(Dataset):
    """
    Create a Dataset for VCTK Corpus, from corresponding CSV files.
    This class emulates torchaudio.datasets.VCTK_092 but with CSV files instead of WAV files.
    """

    def __init__(
        self,
        root: str,
        audio_ext=".csv",
        Fs = 16000,
        resampleMethod = "sinc",
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        self._path = root
        self._txt_dir = os.path.join(self._path, "txt")
        self._audio_dir = os.path.join(self._path, "wav48_silence_trimmed")
        self._audio_ext = audio_ext
        self.resampleMethod = resampleMethod

        if not os.path.isdir(self._path):
            raise RuntimeError("Dataset not found. Please use `download=True` to download it.")

        # Extracting speaker IDs from the folder structure
        self._speaker_ids = sorted(os.listdir(self._txt_dir))
        self._sample_ids = []

        self.Fs = Fs
        self.device = device
        for speaker_id in self._speaker_ids:
            if speaker_id == "p280":
                continue
            utterance_dir = os.path.join(self._txt_dir, speaker_id)
            for utterance_file in sorted(f for f in os.listdir(utterance_dir) if f.endswith(".txt")):
                utterance_id = os.path.splitext(utterance_file)[0]
                audio_path_mic = os.path.join(
                    self._audio_dir,
                    speaker_id,
                    f"{utterance_id}_mic2.csv",
                )
                if speaker_id == "p362" and not os.path.isfile(audio_path_mic):
                    continue
                self._sample_ids.append(utterance_id.split("_"))

    def _load_text(self, file_path) -> str:
        with open(file_path) as file_path:
            return file_path.readlines()[0]

    def _load_audio(self, file_path) -> Tuple[Tensor, int]:
        """
        Load the csv files by running the preprocessing pipeline from core.signal.preprocess
        """
        T, X, Y = processFromFile(file_path, Fs = self.Fs, method = self.resampleMethod, device = self.device)
        XY = torch.stack((X, Y), dim=0)
        
        return XY, self.Fs

    def _load_sample(self, speaker_id: str, utterance_id: str) -> SampleType:
        transcript_path = os.path.join(self._txt_dir, speaker_id, f"{speaker_id}_{utterance_id}.txt")
        audio_path = os.path.join(
            self._audio_dir,
            speaker_id,
            f"{speaker_id}_{utterance_id}_mic2.csv",
        )

        # Reading text
        transcript = self._load_text(transcript_path)

        # Reading FLAC
        waveform, sample_rate = self._load_audio(audio_path)

        return (waveform, sample_rate, transcript, speaker_id, utterance_id)

    def __getitem__(self, n: int) -> SampleType:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            Tuple of the following items;

            Tensor:
                Waveform
            int:
                Sample rate
            str:
                Transcript
            str:
                Speaker ID
            std:
                Utterance ID
        """
        speaker_id, utterance_id = self._sample_ids[n]
        return self._load_sample(speaker_id, utterance_id)

    def __len__(self) -> int:
        return len(self._sample_ids)


class PairedAudioDataset(torch.utils.data.Dataset):
    """
    Create a Dataset for paired data, from two datasets.
    """
    def __init__(self, groundTruthDatasetArgs, sensorDatasetArgs):
        super(PairedAudioDataset, self).__init__()
        # Initialize the dataset with two sub-datasets, an offset and a lowpass filter value.
        self.gtDS = torchaudio.datasets.VCTK_092(**groundTruthDatasetArgs)
        self.sensorDS = VCTK_CSV(**sensorDatasetArgs)
        logging.info("Initializing PairedAudioDataset...")
        # Ensure both datasets have the same length.
        assert len(self.gtDS) == len(self.sensorDS), "Datasets must be the same length"

    def __len__(self):
        # Return the length of the dataset.
        return len(self.gtDS)

    def __getitem__(self, idx):
        # Retrieve items from both datasets.
        wav_gt, sr_gt, txt_gt, speaker_gt, data_gt = self.gtDS[idx]
        wav_sensor, sr_sensor, _, _, _ = self.sensorDS[idx]
        
        if sr_gt != sr_sensor: # Resample if necessary (which is most of the time)
            transform = torchaudio.transforms.Resample(sr_gt, sr_sensor) # The assumption is that srA is the wanted sample rate
            wav_gt = transform(wav_gt)
            
        
        # Return a tuple containing the processed data.
        newTuple = (wav_gt, wav_sensor, sr_sensor, txt_gt, speaker_gt, data_gt)
        return newTuple

def removeOutliers(paired_ds, Fs = 16000):
    if os.path.exists("L.pt"):
        L = torch.load("L.pt")
    else:
        L = []
        for Yorig, Xorig, _, _, _, _ in tqdm(paired_ds):
            Bs = Xorig.shape[-1]
            As = Yorig.shape[-1]
            L.append(Bs - As - 2 * Fs)
        L = torch.tensor(L, dtype=torch.float32)
        torch.save(L, "L.pt")
    # remove outliers from L
    time_threshold = 2 # seconds
    idxx = torch.where(L < Fs * time_threshold)[0] # Mark outliers where the mouse track's length does not match the ground truth
    filtered_paired_ds = torch.utils.data.Subset(paired_ds, idxx)
    Lsubset = L[idxx]
    print(f"Removed {len(L) - len(idxx)} outliers {100*(len(L) - len(idxx))/len(L):.2f}% of the dataset")
    print(f"Remaining {len(paired_ds)} samples")
    return filtered_paired_ds


def getAudioLoaders(filtered_paired_ds, batch_size = 32, split = (0.85, 0.125, 0.025), device = "cpu", Fs = 16000, ret_labels = False):
    # Split into train, test, and dev
    trainSplit, testSplit, devSplit = split
    assert trainSplit + testSplit + devSplit == 1
    
    lenTrain = int(len(filtered_paired_ds) * trainSplit)
    lenTest = int(len(filtered_paired_ds) * testSplit)
    lenDev = len(filtered_paired_ds) - lenTrain - lenTest

    idxtrain = np.random.choice(len(filtered_paired_ds), lenTrain, replace=False)
    idxTest = np.setdiff1d(np.arange(len(filtered_paired_ds)), idxtrain)
    idxtest = np.random.choice(idxTest, lenTest, replace=False)
    idxdev = np.setdiff1d(idxTest, idxtest)
    assert len(idxtrain) + len(idxtest) + len(idxdev) == len(filtered_paired_ds)

    train = torch.utils.data.Subset(filtered_paired_ds, idxtrain)
    test = torch.utils.data.Subset(filtered_paired_ds, idxtest)
    dev = torch.utils.data.Subset(filtered_paired_ds, idxdev)

    # T = 5 * Fs
    T = 501
    n_mels = 80
    def collate(batch):
        N = len(batch)
        out_wavs = torch.zeros((N, 5 * Fs), device=device)
        out_mouse = torch.zeros((N, 2, 5 * Fs), device=device)
        out_utterance = []
        out_speaker = []
        i = 0
        for e in batch:
            W = trim_or_pad(e[0].squeeze(), 5 * Fs).to(device)
            M = trim_or_pad2(e[1][:, Fs:-Fs], 5 * Fs).to(device)

            out_wavs[i] = W
            out_mouse[i] = M

            out_utterance.append(e[5])
            out_speaker.append(e[4])

            i += 1
        if ret_labels:
            return out_mouse, out_wavs, out_speaker, out_utterance
        return out_mouse, out_wavs


    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    dev_loader = torch.utils.data.DataLoader(dev, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    print(f"Train: {len(train_loader)} batches = {len(train_loader) * batch_size} samples")
    print(f"Test: {len(test_loader)} batches = {len(test_loader) * batch_size} samples")
    print(f"Dev: {len(dev_loader)} batches = {len(dev_loader) * batch_size} samples")

    return train_loader, test_loader, dev_loader