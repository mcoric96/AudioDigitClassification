import torch
from torch.utils.data import Dataset


class AudioDigitDataset(Dataset):
    """A PyTorch Dataset for audio digit classification tasks.
    This dataset stores pairs of audio waveforms and their corresponding digit labels.
    Each item in the dataset is a tuple containing a waveform (as a torch.Tensor) and an integer label.
        data (list[tuple[torch.Tensor, int]]):
            A list of tuples, where each tuple consists of a waveform tensor and its associated digit label.
    Example:
        >>> dataset = AudioDigitDataset(data)
        >>> waveform, label = dataset[0]
    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Retrieves the waveform and label at the specified index.
    """

    def __init__(self, data: list[tuple[torch.Tensor, int]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves the waveform and label at the specified index.
        Args:
            idx (int): Index of the data sample to retrieve.
        Returns:
            tuple: A tuple containing the waveform (audio data) and its corresponding label.
        """

        waveform, label = self.data[idx]
        return waveform, label
