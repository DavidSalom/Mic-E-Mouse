from core.nn.AudioMNIST import AudioMnistDataset
from datasets import load_metric, Dataset, DatasetDict, Audio
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer, AutoFeatureExtractor
import torch
import sys

run_name = sys.argv[2]
mouse_only = sys.argv[3]
mouse_only = (mouse_only == "1")

device = "cuda" if torch.cuda.is_available() else "cpu"
wav_dir = "/media/data/AudioMNIST"
csv_dir = sys.argv[1]
MAX_DURATION = 1.5

model_checkpoint = "facebook/wav2vec2-base"

mouse_model_digit = AutoModelForAudioClassification.from_pretrained(model_checkpoint, num_labels=10).to(device)
mouse_model_speaker = AutoModelForAudioClassification.from_pretrained(model_checkpoint, num_labels=61).to(device)
if not mouse_only:
    wavef_model_digit = AutoModelForAudioClassification.from_pretrained(model_checkpoint, num_labels=10).to(device)     
    wavef_model_speaker = AutoModelForAudioClassification.from_pretrained(model_checkpoint, num_labels=61).to(device)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
dataset_pt = AudioMnistDataset(wav_dir, csv_dir, device=device)
datasetDict = dataset_pt.split_dataset()
metric = load_metric("accuracy")

def new_forward(self, input_values):
        hidden_states = input_values

        # make sure hidden_states require grad for gradient_checkpointing
        if self._requires_grad and self.training:
            hidden_states.requires_grad = True

        for conv_layer in self.conv_layers:
            if self._requires_grad and self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    conv_layer.__call__,
                    hidden_states,
                )
            else:
                hidden_states = conv_layer(hidden_states)

        return hidden_states
mouse_model_digit.wav2vec2.feature_extractor.forward = new_forward.__get__(mouse_model_digit.wav2vec2.feature_extractor, mouse_model_digit.wav2vec2.feature_extractor.__class__)
mouse_model_digit.wav2vec2.feature_extractor.conv_layers[0].conv = torch.nn.Conv1d(2, 512, kernel_size=(10,), stride=(5,), bias=False).to(device)

mouse_model_speaker.wav2vec2.feature_extractor.forward = new_forward.__get__(mouse_model_speaker.wav2vec2.feature_extractor, mouse_model_speaker.wav2vec2.feature_extractor.__class__)
mouse_model_speaker.wav2vec2.feature_extractor.conv_layers[0].conv = torch.nn.Conv1d(2, 512, kernel_size=(10,), stride=(5,), bias=False).to(device)

# keep only the waveform_audio and digit_target columns
def preprocess_function(examples):
    return feature_extractor(
        examples, 
        padding='max_length',
        sampling_rate=feature_extractor.sampling_rate, 
        max_length=int(feature_extractor.sampling_rate * MAX_DURATION), 
        truncation=True, 
    )

def preprocess_function_mouse(examples):
    return feature_extractor(
        examples, 
        padding='max_length',
        sampling_rate=feature_extractor.sampling_rate, 
        max_length=int(feature_extractor.sampling_rate * MAX_DURATION), 
        truncation=True, 
    )

if not mouse_only:
    WV_DG_dataset = datasetDict.map(preprocess_function, input_columns=["original_audio"], batched=True).remove_columns(["mouse_audio", "speaker_target", "original_audio"]).rename_column("digit_target", "labels")

    WV_SP_dataset = datasetDict.map(preprocess_function, input_columns=["original_audio"], batched=True).remove_columns(["mouse_audio", "digit_target", "original_audio"]).rename_column("speaker_target", "labels")

MS_DG_dataset = datasetDict.map(preprocess_function_mouse, input_columns=["mouse_audio"], batched=False).remove_columns(["original_audio", "speaker_target", "mouse_audio"]).rename_column("digit_target", "labels")

MS_SP_dataset = datasetDict.map(preprocess_function_mouse, input_columns=["mouse_audio"], batched=False).remove_columns(["original_audio", "digit_target", "mouse_audio"]).rename_column("speaker_target", "labels")

args = TrainingArguments(
    "test",
    evaluation_strategy = "epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=64,
    num_train_epochs=10,
    warmup_ratio=0.1,
    logging_steps=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
    remove_unused_columns=False
)

import numpy as np

def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

MS_DG_TRAINER = Trainer(
    mouse_model_digit,
    args,
    train_dataset=MS_DG_dataset["train"],
    eval_dataset=MS_DG_dataset["test"],
    compute_metrics=compute_metrics
)

MS_SP_TRAINER = Trainer(
    mouse_model_speaker,
    args,
    train_dataset=MS_SP_dataset["train"],
    eval_dataset=MS_SP_dataset["test"],
    compute_metrics=compute_metrics
)
if not mouse_only:
    WV_DG_TRAINER = Trainer(
        wavef_model_digit,
        args,
        train_dataset=WV_DG_dataset["train"],
        eval_dataset=WV_DG_dataset["test"],
        compute_metrics=compute_metrics
    )

    WV_SP_TRAINER = Trainer(
        wavef_model_speaker,
        args,
        train_dataset=WV_SP_dataset["train"],
        eval_dataset=WV_SP_dataset["test"],
        compute_metrics=compute_metrics
    )


MS_DG_TRAINER.train()
MS_SP_TRAINER.train()
if not mouse_only:
    WV_DG_TRAINER.train()
    WV_SP_TRAINER.train()

# Get accuracy
MS_DG_ACC = MS_DG_TRAINER.evaluate()['eval_accuracy']
MS_SP_ACC = MS_SP_TRAINER.evaluate()['eval_accuracy']
if not mouse_only:
    WV_DG_ACC = WV_DG_TRAINER.evaluate()['eval_accuracy']
    WV_SP_ACC = WV_SP_TRAINER.evaluate()['eval_accuracy']

print(f"MS_DG_ACC: {MS_DG_ACC}")
print(f"MS_SP_ACC: {MS_SP_ACC}")
if not mouse_only:
    print(f"WV_DG_ACC: {WV_DG_ACC}")
    print(f"WV_SP_ACC: {WV_SP_ACC}")

# Save the models
mouse_model_digit.save_pretrained(f"/media/data/models/{run_name}/mouse_model_digit")
mouse_model_speaker.save_pretrained(f"/media/data/models/{run_name}/mouse_model_speaker")
if not mouse_only:
    wavef_model_digit.save_pretrained(f"/media/data/models/{run_name}/wavef_model_digit")
    wavef_model_speaker.save_pretrained(f"/media/data/models/{run_name}/wavef_model_speaker")