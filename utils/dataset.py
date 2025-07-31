import torch
from torch.utils.data import Dataset

class SessionDataset(Dataset):
    """
    Simple PyTorch Dataset wrapper for embedding data.
    Assumes input is a tensor or NumPy array of shape (N, F).
    """
    def __init__(self, data_tensor):
        if isinstance(data_tensor, list):
            data_tensor = torch.tensor(data_tensor, dtype=torch.float32)
        self.data = data_tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
