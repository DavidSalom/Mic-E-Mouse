import torch
import torch.nn as nn
from torchinfo import summary
from core.VCTK_dataloader import VCTK_Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import hashlib
from core.WaveUNet import *
from einops import*
import logging

logging.basicConfig(level=logging.DEBUG)

train_split = 0.9
val_split = 0.05
test_split = 0.05
batch_size = 16
resample_rate = 8000
VCTK_root_path = "/media/data/VCTK-Corpus"

DS = VCTK_Dataset.cache_constructor(VCTK_root_path, resample_rate=resample_rate)
logging.info(f"Splitting the dataset into {train_split * 100}% training, {val_split * 100}% validation, and {test_split * 100}% test sets.")
N_train = int(len(DS) * train_split)
N_val = int(len(DS) * val_split)
N_test = len(DS) - N_train - N_val
logging.info(f"Number of training samples: {N_train}; number of validation samples: {N_val}; number of test samples: {N_test}.")
# Split the dataset into training, validation, and test sets
train_set, val_set, test_set = torch.utils.data.random_split(DS, [N_train, N_val, N_test], generator=torch.Generator().manual_seed(42))
# Create data loaders
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4)

def compute_statistics_hardcoded():
    val = torch.concatenate([X for X, _ in val_loader], dim=0)
    val = ( val - val.mean(dim=-1, keepdim=True) ) / val.std(dim=-1, keepdim=True)
    clip_max = val.max(dim = -1)[0].quantile(0.95)
    clip_min = val.min(dim = -1)[0].quantile(0.05)
    # val = torch.clip(val, clip_min, clip_max)
    # val = 2 * (val - clip_min) / (clip_max - clip_min) - 1
    return clip_min, clip_max