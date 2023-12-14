"""
Flow Matching in Latent space
"""

import argparse
import os
import shutil
from functools import partial
from time import time

import torch
from omegaconf import OmegaConf

import torch.nn.functional as F
import torch.optim as optim
import torchvision
from accelerate import Accelerator
from accelerate.utils import set_seed


from synthgen_pytorch.data.datasets_prep import get_dataset
from EMA import EMA
from models import create_network
from torchdiffeq import odeint_adjoint as odeint

# faster training
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True