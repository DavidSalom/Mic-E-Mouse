import os
from typing import Tuple

import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
from ..signal.preprocess import *
from ..signal.align import *
from .utils import CacheMixin
from ..cache import *
import logging
import hashlib

SampleType = Tuple[Tensor, int, str, str, str]


class VCTK_CSV(Dataset):
    """
    Create a Dataset for VCTK Corpus, from corresponding CSV files.
    This class emulates torchaudio.datasets.VCTK_092 but with CSV files instead of WAV files.
    """

    def __init__(
        self,
        root: str,
        audio_ext=".csv"
    ):
        self._path = root
        self._txt_dir = os.path.join(self._path, "txt")
        self._audio_dir = os.path.join(self._path, "wav48_silence_trimmed")
        self._audio_ext = audio_ext

        if not os.path.isdir(self._path):
            raise RuntimeError("Dataset not found. Please use `download=True` to download it.")

        # Extracting speaker IDs from the folder structure
        self._speaker_ids = sorted(os.listdir(self._txt_dir))
        self._sample_ids = []

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
        T, X, Y = processFromFile(file_path, skipPCA = True)
        XY = torch.stack((X, Y), dim=0)
        
        return XY, 16000

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
    def __init__(self, groundTruthDataset, sensorDataset):
        super(PairedAudioDataset, self).__init__()
        # Initialize the dataset with two sub-datasets, an offset and a lowpass filter value.
        self.gtDS = groundTruthDataset
        self.sensorDS = sensorDataset
        logging.info("Initializing PairedAudioDataset...")
        # Ensure both datasets have the same length.
        assert len(groundTruthDataset) == len(sensorDataset), "Datasets must be the same length"

    def __len__(self):
        # Return the length of the dataset.
        return len(self.gtDS)

    def __getitem__(self, idx):
        # Retrieve items from both datasets.
        wav_gt, sr_gt, txt_gt, speaker_gt, data_gt = self.gtDS[idx]
        wav_sensor, sr_sensor, _, _, _ = self.sensorDS[idx]
        
        if sr_gt != sr_sensor: # Resample if necessary (which is most of the time)
            transform = torchaudio.transforms.Resample(sr_sensor, sr_gt) # The assumption is that srA is the wanted sample rate
            wav_sensor = transform(wav_sensor)
            
        
        # Return a tuple containing the processed data.
        newTuple = (wav_gt, wav_sensor, sr_sensor, txt_gt, speaker_gt, data_gt)
        return newTuple
