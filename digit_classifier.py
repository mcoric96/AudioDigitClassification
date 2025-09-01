import json
from pathlib import Path

import pytorch_lightning as pl
from audio_digit_dataset import AudioDigitDataset
from audio_model import AudioTransformerClassifier
from torch.utils.data import DataLoader
from utils import (
    extract_audio_list,
    pad_or_truncate_audio_list,
    read_parquet_file,
    train_val_split,
)

if __name__ == "__main__":
    # load settings for model, data and training
    with open("settings.json", "r") as f:
        settings = json.load(f)

    # load train and test data
    train_file = read_parquet_file(Path("data", "train-00000-of-00001.parquet"))
    test_file = read_parquet_file(Path("data", "test-00000-of-00001.parquet"))

    # obtain data as list of tuples (waveform, label)
    audio_list = extract_audio_list(train_file)
    test_audio_list = extract_audio_list(test_file)

    # need to otain validation set from training data
    train_audio_list, val_audio_list = train_val_split(
        audio_list, val_ratio=settings["data"]["val_ratio"], random_seed=42
    )

    # used for padding / truncating all audio signals to the same length
    TARGET_LENGTH = max(
        [audio_list_object[0].shape[0] for audio_list_object in train_audio_list]
    )

    train_audio_list = pad_or_truncate_audio_list(train_audio_list, TARGET_LENGTH)
    val_audio_list = pad_or_truncate_audio_list(val_audio_list, TARGET_LENGTH)
    test_audio_list = pad_or_truncate_audio_list(test_audio_list, TARGET_LENGTH)

    # create datasets and dataloaders
    train_dataset = AudioDigitDataset(train_audio_list)
    val_dataset = AudioDigitDataset(val_audio_list)
    test_dataset = AudioDigitDataset(test_audio_list)

    BATCH_SIZE = settings["training"]["batch_size"]
    NUM_WORKERS = settings["training"]["num_workers"]

    # create dataloaders for training, validation and testing
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    audio_digit_model = AudioTransformerClassifier(
        input_length=TARGET_LENGTH,
        num_input_features=1,
        num_classes=10,
        d_model=settings["model"]["d_model"],
        nhead=settings["model"]["nhead"],
        num_encoder_layers=settings["model"]["num_encoder_layers"],
        dim_feedforward=settings["model"]["dim_feedforward"],
        dropout=settings["model"]["dropout"],
        pooling_mode=settings["model"]["pooling_mode"],
    )

    MAX_EPOCHS = settings["training"]["max_epochs"]
    EARLY_STOPPING_PATIENCE = settings["training"]["early_stopping_patience"]

    # model callbacks for early stopping and model checkpointing during training
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, mode="min", verbose=True
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss", mode="min", save_top_k=1, filename="best-checkpoint"
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator=settings["training"]["trainer_accelerator"],
        callbacks=[early_stop_callback, checkpoint_callback],
    )

    # after training, load the best checkpoint
    trainer.fit(audio_digit_model, train_dataloader, val_dataloader)
    best_model_path = checkpoint_callback.best_model_path
    audio_digit_model = AudioTransformerClassifier.load_from_checkpoint(best_model_path)
