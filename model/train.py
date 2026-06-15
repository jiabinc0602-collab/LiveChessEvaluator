"""
This script is used to train the model
"""
import torch
import pathlib
import glob
from torch.utils.data import Dataset


class ChessDataset(Dataset):
    def __init__(self):
        tensor_path = pathlib.Path("data/tensors").glob("*.pt")
        all_X = []
        all_Y = []
        # appending all X and Y values from tensors
        for tensor in tensor_path:
            data = torch.load(tensor, map_location='cpu')
            all_X.append(data["X"])
            all_Y.append(data["Y"])
        self.X = torch.cat(all_X, dim = 0)
        self.Y = torch.cat(all_Y, dim = 0)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
