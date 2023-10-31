import os
from typing import Tuple

import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
from ..signal.preprocess import *

SampleType = Tuple[Tensor, int, str, str, str]


class VCTK_092(Dataset):
    """*VCTK 0.92* :cite:`yamagishi2019vctk` dataset

    Args:
        root (str): Root directory where the dataset's top level directory is found.
        mic_id (str, optional): Microphone ID. Either ``"mic1"`` or ``"mic2"``. (default: ``"mic2"``)
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
        audio_ext (str, optional): Custom audio extension if dataset is converted to non-default audio format.

    Note:
        * All the speeches from speaker ``p315`` will be skipped due to the lack of the corresponding text files.
        * All the speeches from ``p280`` will be skipped for ``mic_id="mic2"`` due to the lack of the audio files.
        * Some of the speeches from speaker ``p362`` will be skipped due to the lack of  the audio files.
        * See Also: https://datashare.is.ed.ac.uk/handle/10283/3443
    """

    def __init__(
        self,
        root: str,
        mic_id: str = "mic2",
        audio_ext=".csv",
        maxLen = 226209
    ):
        if mic_id not in ["mic1", "mic2"]:
            raise RuntimeError(f'`mic_id` has to be either "mic1" or "mic2". Found: {mic_id}')

        self._path = root
        self._txt_dir = os.path.join(self._path, "txt")
        self._audio_dir = os.path.join(self._path, "wav48_silence_trimmed")
        self._mic_id = mic_id
        self._audio_ext = audio_ext
        self.maxLen = maxLen

        if not os.path.isdir(self._path):
            raise RuntimeError("Dataset not found. Please use `download=True` to download it.")

        # Extracting speaker IDs from the folder structure
        self._speaker_ids = sorted(os.listdir(self._txt_dir))
        self._sample_ids = []

        """
        Due to some insufficient data complexity in the 0.92 version of this dataset,
        we start traversing the audio folder structure in accordance with the text folder.
        As some of the audio files are missing of either ``mic_1`` or ``mic_2`` but the
        text is present for the same, we first check for the existence of the audio file
        before adding it to the ``sample_ids`` list.

        Once the ``audio_ids`` are loaded into memory we can quickly access the list for
        different parameters required by the user.
        """
        for speaker_id in self._speaker_ids:
            if speaker_id == "p280" and mic_id == "mic2":
                continue
            utterance_dir = os.path.join(self._txt_dir, speaker_id)
            for utterance_file in sorted(f for f in os.listdir(utterance_dir) if f.endswith(".txt")):
                utterance_id = os.path.splitext(utterance_file)[0]
                audio_path_mic = os.path.join(
                    self._audio_dir,
                    speaker_id,
                    f"{utterance_id}_{mic_id}{self._audio_ext}",
                )
                if speaker_id == "p362" and not os.path.isfile(audio_path_mic):
                    continue
                self._sample_ids.append(utterance_id.split("_"))

    def _load_text(self, file_path) -> str:
        with open(file_path) as file_path:
            return file_path.readlines()[0]

    def _load_audio(self, file_path) -> Tuple[Tensor, int]:
        T, X, Y = processFromFile(file_path, skipPCA = True)
        X = torch.nn.functional.pad(X, (0, self.maxLen - X.shape[0]))
        Y = torch.nn.functional.pad(Y, (0, self.maxLen - Y.shape[0]))
        XY = torch.stack((X, Y), dim=0)
        
        return XY, 16000

    def _load_sample(self, speaker_id: str, utterance_id: str, mic_id: str) -> SampleType:
        transcript_path = os.path.join(self._txt_dir, speaker_id, f"{speaker_id}_{utterance_id}.txt")
        audio_path = os.path.join(
            self._audio_dir,
            speaker_id,
            f"{speaker_id}_{utterance_id}_{mic_id}{self._audio_ext}",
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
        return self._load_sample(speaker_id, utterance_id, self._mic_id)

    def __len__(self) -> int:
        return len(self._sample_ids)
