from torchsummary import summary as summary
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image as PILImage
import torch
import torchvision
from torchvision import transforms 
import glob
import random
import cv2
from random import shuffle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
import torch.nn.functional as F


device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)
if device == 'cuda':
  torch.cuda.manual_seed_all(42)

unet = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True, scale=0.5).to(device)

model = unet.to(device)
model.n_classes
unet.modules
unet.n_classes = 1
unet.outc = nn.Conv2d(64, 1, kernel_size=1).to(device)
summary(unet, (3, 224, 224))