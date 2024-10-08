# Setup the imports
import core.nn.AudioMNIST
from core.nn.whisperPrimitives import *
from torch.utils.data import DataLoader
from core.nn.utils import *
import argparse
from tqdm import tqdm
import torch.nn as nn
import einops


Fs = 16000
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create the argument parser
parser = argparse.ArgumentParser(description='AudioMNIST Speaker Recognition')
parser.add_argument('--dataset_path', type=str, default="/media/AICPS/MICEMOUSE/AudioMNIST/gen/csv", help='Path to the dataset')

# Parse the arguments
args = parser.parse_args()

# Load the dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ds = core.nn.AudioMNIST.AudioMnistDataset(wav_basedir="/media/AICPS/MICEMOUSE/AudioMNIST", csv_basedir=args.dataset_path, device = device, Fs=Fs)
train_loader, test_loader, dev_loader = core.nn.AudioMNIST.get_audio_loaders(ds, device = device, batch_size=32, ret_labels=True)

tformWrapper = core.nn.utils.transformWrapper(Fs = Fs, 
                                              n_fft = 800,
                                              win_length = 800, 
                                              hop_length = 160, 
                                              n_mels = 80,
                                              f_max_mouse = 8000, 
                                              f_max_full = 8000).to(device)

A = next(iter(train_loader))

tformWrapper.computeMinMax(dev_loader)

N_EPOCHS = 3

# Define the model
nDigits = 10
nSpeakersMNIST = 61
digitList = list(range(10)) # Very hacky
speakerListMNIST = list(range(nSpeakersMNIST))

MNIST_GT_SPEAKER = buildClassifier(80, nDigits, 101, 352, device)
MNIST_MS_SPEAKER = buildClassifier(160, nSpeakersMNIST, 101, 352, device)
MNIST_MS_UTTERANCE = buildClassifier(160, nDigits, 101, 352, device)

torch.compile(MNIST_GT_SPEAKER)
torch.compile(MNIST_MS_SPEAKER)
torch.compile(MNIST_MS_UTTERANCE)

train(MNIST_GT_SPEAKER, "GT", "UTTERANCE", train_loader, digitList, n_epochs=50, tf_wrapper=tformWrapper, device=device)
# train(MNIST_MS_SPEAKER, "MS", "SPEAKER", train_loader, speakerListMNIST, n_epochs=50, tf_wrapper=tformWrapper, device=device)
# train(MNIST_MS_UTTERANCE, "MS", "UTTERANCE", train_loader, digitList, n_epochs=50, tf_wrapper=tformWrapper, device=device)

# Save the models
model_name = f"models/{args.dataset_path.split('/')[-1]}_MNIST_MS_SPEAKER.model"
torch.save(MNIST_MS_SPEAKER.state_dict(), model_name)
print(f"Saved model to {model_name}")

model_name = f"models/{args.dataset_path.split('/')[-1]}_MNIST_MS_UTTERANCE.model"
torch.save(MNIST_MS_UTTERANCE.state_dict(), model_name)
print(f"Saved model to {model_name}")

# Get the accuracy
MNIST_MS_SPEAKER_ACC = getAccuracy(MNIST_MS_SPEAKER, "MS", "SPEAKER", test_loader, speakerListMNIST, tformWrapper, device)
print(f"MS Speaker Accuracy: {MNIST_MS_SPEAKER_ACC}")

MNIST_MS_UTTERANCE_ACC = getAccuracy(MNIST_MS_UTTERANCE, "MS", "UTTERANCE", test_loader, digitList, tformWrapper, device)
print(f"MS Utterance Accuracy: {MNIST_MS_UTTERANCE_ACC}")

# train(MNIST_MS_UTTERANCE, "MS", "UTTERANCE", train_loader, digitList, n_epochs=20)

# MNIST_MS_SPEAKER_ACC = getAccuracy(MNIST_MS_SPEAKER, "MS", "SPEAKER", dataloader, speakerListMNIST)
# MNIST_MS_UTTERANCE_ACC = getAccuracy(MNIST_MS_UTTERANCE, "MS", "UTTERANCE", dataloader, digitList)
# print(f"MS Speaker Accuracy: {MNIST_MS_SPEAKER_ACC}")
# print(f"MS Utterance Accuracy: {MNIST_MS_UTTERANCE_ACC}")

# def getRes(datasetPath):
#     ds = core.nn.AudioMNIST.audioMnistDataset(basedir=datasetPath, device = device, Fs=Fs)
#     train_loader, test_loader, dev_loader = core.nn.AudioMNIST.getAudioLoaders(ds, device = device, batch_size=32, ret_labels=True)
#     nDigits = 10
#     nSpeakersMNIST = 61
#     digitList = list(range(10)) # Very hacky
#     speakerListMNIST = list(range(nSpeakersMNIST))
    
#     MNIST_MS_SPEAKER = buildClassifier(160, nSpeakersMNIST, 101, 352)
#     MNIST_MS_UTTERANCE = buildClassifier(160, nDigits, 101, 352)

#     # MNIST_MS_SPEAKER.load_state_dict(torch.load("models/MNIST_MS_SPEAKER.model"))
#     train(MNIST_MS_SPEAKER, "MS", "SPEAKER", train_loader, speakerListMNIST, n_epochs=20)

#     # MNIST_MS_UTTERANCE.load_state_dict(torch.load("models/MNIST_MS_UTTERANCE.model"))
#     train(MNIST_MS_UTTERANCE, "MS", "UTTERANCE", train_loader, digitList, n_epochs=20)

#     MNIST_MS_SPEAKER_ACC = getAccuracy(MNIST_MS_SPEAKER, "MS", "SPEAKER", test_loader, speakerListMNIST)
#     MNIST_MS_UTTERANCE_ACC = getAccuracy(MNIST_MS_UTTERANCE, "MS", "UTTERANCE", test_loader, digitList)

#     print(f"For Dataset: {datasetPath}")
#     print(f"MS Speaker Accuracy: {MNIST_MS_SPEAKER_ACC}")
#     print(f"MS Utterance Accuracy: {MNIST_MS_UTTERANCE_ACC}")

# for fn in os.listdir("/media/result/"):
#     getRes(f"/media/result/{fn}")