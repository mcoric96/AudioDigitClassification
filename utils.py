import io
import random

import pandas as pd
import soundfile as sf
import torch


def read_parquet_file(file_path):
    """
    Reads a Parquet file and returns a pandas DataFrame.

    Args:
        file_path (str): Path to the Parquet file.

    Returns:
        pd.DataFrame: DataFrame containing the data from the Parquet file.
    """
    return pd.read_parquet(file_path)


def pad_or_truncate(waveform: torch.Tensor, target_length: int) -> torch.Tensor:
    """
    Pads or truncates a waveform tensor to a specified target length.
    If the waveform is longer than the target length, it is truncated.
    If the waveform is shorter, it is padded with zeros at the end.
    Args:
        waveform (torch.Tensor): The input waveform tensor. Assumes the last dimension is the time axis.
        target_length (int): The desired length of the output waveform.
    Returns:
        torch.Tensor: The waveform tensor padded or truncated to the target length.
    """
    current_length = waveform.shape[0]

    if current_length > target_length:
        waveform = waveform[:target_length, :]  # truncate
    elif current_length < target_length:
        pad_amount = target_length - current_length  # pad with zeros at the end
        waveform = torch.cat([waveform, torch.zeros(pad_amount, 1)], dim=0)

    return waveform


def pad_or_truncate_audio_list(
    audio_list, target_length
) -> list[tuple[torch.Tensor, int]]:
    """
    Pads or truncates all audio signals in audio_list to target_length.

    Args:
        audio_list (list): List of tuples (waveform, label).
        target_length (int): Desired length for all waveforms.

    Returns:
        list[tuple[torch.Tensor, int]]: List of tuples (padded_or_truncated_waveform, label).
    """
    return [
        (pad_or_truncate(waveform, target_length), label)
        for waveform, label in audio_list
    ]


def extract_audio_list(file: pd.DataFrame) -> list:
    """
    Extracts audio waveforms and labels from a DataFrame.

    Args:
        file (pd.DataFrame): DataFrame containing audio bytes and labels.

    Returns:
        list: List of tuples (waveform_tensor, label).
    """
    audio_list = []

    for _, row in file.iterrows():
        audio_bytes = row["audio"]["bytes"]
        label = row["label"]
        audio_buffer = io.BytesIO(audio_bytes)
        audio_data, _ = sf.read(audio_buffer)
        audio_list.append((torch.Tensor(audio_data).unsqueeze(-1), label))

    return audio_list


def train_val_split(
    audio_list: list[tuple[torch.Tensor, int]],
    val_ratio: float = 0.2,
    random_seed: int = 42,
) -> tuple[list, list]:
    """
    Splits the audio list into training and validation sets.

    Args:
        audio_list (list): List of tuples (waveform, label).
        val_ratio (float): Proportion of data to use for validation.
        random_seed (int): Seed for random shuffling.

    Returns:
        tuple: (train_list, val_list)
    """

    random.seed(random_seed)
    random.shuffle(audio_list)
    split_index = int(len(audio_list) * (1 - val_ratio))

    return audio_list[:split_index], audio_list[split_index:]
