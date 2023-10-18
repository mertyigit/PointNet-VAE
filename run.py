'''
Train Validate Evaluation Script
'''

import os
import sys
import re
from glob import glob
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torchmetrics.classification import MulticlassMatthewsCorrCoef
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn.functional import kl_div


import open3d as o3


import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, fps, global_max_pool, radius
from torch_geometric.nn.conv import PointConv

sys.path.append('../src')

from models.utils import PointsTo3DShape
from models.PointNetEncoder import PointNetBackbone
from utils.calculate_loss import ChamferDistanceLoss
from models.PointCloudEncoder import PointCloudEncoder
from models.PointCloudDecoder import PointCloudDecoder, PointCloudDecoderSelf, PointCloudDecoderMLP
from models.AutoEncoder import AutoEncoder
from models.VAE import VAE
from data.dataset import DataModelNet
from utils.utils import *

from tqdm import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline


### HERE ARGS ###
parser = argparse.ArgumentParser(description='Training Testing arguments')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'Path to config file.',
                    default='configs/vae.yaml')

args = parser.parse_args()

#################

### REPRODUCIBILITY ###
seed_everything(config['seed'], True)
#######################


print('MPS is build: {}'.format(torch.backends.mps.is_built()))
print('MPS Availability: {}'.format(torch.backends.mps.is_available()))
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps'
print('Device is set to :{}'.format(DEVICE))

# General parameters
#NUM_TRAIN_POINTS = 8192
NUM_TEST_POINTS = 2048

NUM_POINTS = 1024
NUM_CLASSES = 16

# model hyperparameters
GLOBAL_FEATS = 1024
BATCH_SIZE = 64

EPOCHS = 1000
LR = 0.0001
REG_WEIGHT = 0.001 

LATENT_DIM = 128

#### LOAD DATA ####
data = ModelNet(**config["data_parameters"])
data.setup()
train_dataloader = data.train_dataloader()
val_dataloader = data.val_dataloader()
###################

model_run = Trainer(
        model=vae(),
        criterion=ChamferDistanceLoss(),
        optimizer=optim.Adam(vae.parameters(), lr=0.001),
        encoder_type='ConvolutionEncoder',
        model_type='VAE',
        checkpoint='./checkpoints',
        experiment='trial',
        device='mps',)
    
model_run.fit(train_dataloader, val_dataloader, EPOCHS)












