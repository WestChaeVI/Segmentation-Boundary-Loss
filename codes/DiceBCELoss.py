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

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = torch.sigmoid(inputs)  
        
        # dict형태로 데이터가 들어오는 경우가 있음 ######################################################################
        if isinstance(inputs, dict):
            inputs = torch.sigmoid(inputs['out'])
        else:
            inputs = torch.sigmoid(inputs)       # 픽셀들은 0~1 값
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)  # flatten된 tensor화
        inputs_ = inputs > 0.5    # True(1)/False(0)로 boolean값 도출

        targets = targets.view(-1) # flatten된 tensor화
        targets[targets != 0] = 1  # 0~255 값을 가지기 때문에 ----> 0이 아닌 것들은 1로
        
        intersection = (inputs_ * targets).sum()    # 교집합 구하기                        
        dice_loss = 1 - (2.*intersection + smooth)/(inputs_.sum() + targets.sum() + smooth)  # 1 - (2 x (교집합+smooth)/(input.sum + target.sum + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')  # BCE는 0 또는 1 값 추출
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE
        
def dice_pytorch_eval(outputs: torch.Tensor, labels: torch.Tensor):

    # comment out if your model contains a sigmoid or equivalent activation layer
    # outputs = torch.sigmoid(outputs)
    
    # dict형태로 데이터가 들어오는 경우가 있음 ######################################################################
    if isinstance(outputs, dict):
        outputs = torch.sigmoid(outputs['out'])
    else:
        outputs = torch.sigmoid(outputs)  
    
    
    # thresholding since that's how we will make predictions on new imputs
    outputs = outputs > 0.5

    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1).byte()  # (BATCH, 1, H, W) -> (BATCH, H, W)
    labels = labels.squeeze(1).byte()


    SMOOTH = 1e-8
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zero if both are 0
    dice = 2 * (intersection + SMOOTH) / (intersection + union + SMOOTH) # We smooth our devision to avoid 0/0

    return dice.mean()

def iou_pytorch_eval(outputs: torch.Tensor, labels: torch.Tensor):

    # comment out if your model contains a sigmoid or equivalent activation layer
    # outputs = torch.sigmoid(outputs) 
    # dict형태로 데이터가 들어오는 경우가 있음 ######################################################################
    if isinstance(outputs, dict):
        outputs = torch.sigmoid(outputs['out'])
    else:
        outputs = torch.sigmoid(outputs)          

    # thresholding since that's how we will make predictions on new imputs
    outputs = outputs > 0.5

    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1).byte()  # (BATCH, 1, H, W) -> (BATCH, H, W)
    labels = labels.squeeze(1).byte()


    SMOOTH = 1e-8
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zero if both are 0
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    return iou.mean()