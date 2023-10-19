import os
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import zipfile

import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, fps, global_max_pool, radius
from torch_geometric.nn.conv import PointConv


# Custom Data Class for future applications
class MyDataset(Dataset):
    def __init__(self):
        pass
    
    
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass


class DataModelNet():
    """
    PyTorch Lightning data module 

    Args:
        data_dir:
        train_batch_size: 
        val_batch_size: 
        patch_size: 
        num_workers:
        pin_memory:
    """

    def __init__(
        self,
        data_path: str,
        pre_transform,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        train_num_points: int = 1024,
        val_num_points: int = 1024,
        **kwargs,
    ):
        super().__init__()

        self.data_path = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.train_num_points = train_num_points
        self.val_num_points = val_num_points
        self.pre_transform = pre_transform

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = ModelNet(
                            root=self.data_path,
                            name=self.data_path[-2:],
                            train=True,
                            pre_transform=self.pre_transform,
                            transform=T.SamplePoints(self.train_num_points),
        )
        
        self.val_dataset = ModelNet(
                            root=self.data_path,
                            name=self.data_path[-2:],
                            train=False,
                            pre_transform=self.pre_transform,
                            transform=T.SamplePoints(self.val_num_points),
        )
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
                    self.train_dataset,
                    batch_size=self.train_batch_size,
                    shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
                    self.val_dataset,
                    batch_size=self.val_batch_size,
                    shuffle=False,
        )
    
