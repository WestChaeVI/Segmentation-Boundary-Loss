import glob
import os
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import numpy as np
import tensorflow
import cv2

images = sorted(glob.glob('/content/drive/MyDrive/Segmentation/images/*'))
labels = sorted(glob.glob('/content/drive/MyDrive/Segmentation/labels/*'))
PILImage.open(images[0])
PILImage.open(labels[0])


fig = plt.figure(figsize=(20,10))
plt = fig.subplots(2,1)

plt[0].axis('off')

img = cv2.imread(images[0])
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
mask = cv2.imread(labels[0])



# 마스킹을 보여주기 위해 흰색처리
real_mask = mask.copy()

# 마스크 씌운 이미지를 보여주기 위한 처리
mask[mask == 0] = 0
mask[mask != 0 ] = 1
masked_img = img * mask


#이미지/ 라벨(마스크)/ 마스크를 씌운 이미지
plt[0].imshow( np.concatenate([img, real_mask, masked_img], axis = 1) )


# ----------------------------------------------------------------------

import seaborn as sns

# 해당 데이터셋의 라벨의 픽셀값 -> 0: 배경 , 250: 객체(나비)
mask = cv2.imread(labels[0])
plt[1] = sns
_=plt[1].distplot(mask)