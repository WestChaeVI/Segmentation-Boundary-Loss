import os
import glob
import cv2
import random
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms, datasets
from Codes.dataset import WestChaeVI_Dataset


_size = 224, 224
resize = transforms.Resize(_size, interpolation=0)

# set your transforms 
train_transforms = transforms.Compose([
                           transforms.Resize(_size, interpolation=0),
                           transforms.RandomRotation(180),
                           transforms.RandomHorizontalFlip(0.5),
                           transforms.RandomCrop(_size, padding = 10), # needed after rotation (with original size)
                       ])

test_transforms = transforms.Compose([
                           transforms.Resize(_size, interpolation=0),
                       ])

batch_size = 8

trainset = WestChaeVI_Dataset(root_path = '/content/drive/MyDrive/Segmentation/', mode = 'train', transforms = train_transforms)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle = True, num_workers = 2)

validset = WestChaeVI_Dataset(root_path = '/content/drive/MyDrive/Segmentation/', mode = 'valid', transforms = test_transforms)
valid_loader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle = True, num_workers = 2)