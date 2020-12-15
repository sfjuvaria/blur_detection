#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 18:46:04 2020

@author: new
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 11:35:47 2020

@author: JS
"""
import cv2
import os, sys
from os.path import exists, join

import numpy as np
import pandas as pd
from PIL import Image
Image.LOAD_TRUNCATED_IMAGES = True

from scipy.ndimage import gaussian_filter, laplace,\
                          convolve, generic_filter
import matplotlib.pyplot as plt                          
import skimage
from skimage import io
from skimage.color import rgb2gray

import torch
from torchvision import datasets
from torch.utils.data import DataLoader

def normalize(x):
  return (x - x.min()) / (x.max() - x.min())

# 3 quality measures, can add more here
def ACMO(im):
  # Absolute Central Moment (Shirvaikar2004)
  (m, n) = im.shape
  hist, _ = np.histogram(im, bins=256, range=(0, 1))
  hist = hist / (m * n)
  hist = abs(np.linspace(0, 1, num=256) - np.mean(im[:])) * hist
  fm = sum(hist)
  return fm

def BREN(im):
  # Brenner's (Santos97)
  (m, n) = im.shape
  dh = np.zeros((m, n))
  dv = np.zeros((m, n))
  dv[0:m-2,:] = im[2:,:] - im[0:-2,:]
  dh[:,0:n-2] = im[:,2:] - im[:,0:-2]
  fm = np.maximum(dh, dv)
  fm = fm**2
  fm = fm.mean()
  return fm

def LAPV(im):
  # Variance of laplacian (Pech2000)
  fm = laplace(im)
  fm = fm.std()**2
  return fm

# Load data directory folder
data_dir = '/home/new/Documents/data_cropped/png/CounterfeitAnalysis/2016-2019CAM_CROP1/bch'



# Save images path in a list
img_list = []
# file_list = []
for roots, _, files in os.walk(data_dir):
    for file in files:
        if file.endswith(".JPG"):
            img_path = os.path.join(roots, file)
            img_list.append(img_path)
            # file_list = file
        if file.endswith(".jpg"):
            img_path = os.path.join(roots, file)
            img_list.append(img_path)
            # file_list = file
# sort on basis of filename not pathlist name- how?

# Read/open image, convert to grayscale
# Find shape, calc size, find min and max
# Save the values in the row of the pd table

s = np.zeros((len(img_list),2), dtype=int)
# Headings of the pd table
# df = pd.DataFrame(columns=['filepath', 'height', 'width', 'size'])

for ind, img_path in enumerate(img_list):   
    im = skimage.io.imread(img_path)
    x, y, channels = im.shape
    s[ind,:] = [x, y]
    
    
size= s[:,0]*s[:,1]
size = np.sort(size)
x = np.arange(len(img_list))
plt.scatter(x,size)
plt.show()


# df = df.append({'filepath':img_path, 'height':y, 'width':x, 'size':x*y}, ignore_index=True)

# # Sort the list based on size of the images
# df_sorted = df.sort_values(by='size', ascending= True)



# # Select 100 sample images
# dfr = df_sorted.sample(n=100)
# # # Sort the list based on size of the images
# dfr_sorted = dfr.sort_values(by='size', ascending= True)
# dfr_index = dfr_sorted.set_index(['size']).reset_index()

# # Plot the graphs-size vs lapv,bren,acmo
# dfr_index.plot(x='size', y='lapv') 
# dfr_index.plot(x='size', y='bren')
# dfr_index.plot(x='size', y='acmo')

# # Convert into series format to plot
# arr_size = df_sorted['size']
# arr_lapv = df_sorted['lapv']

# ts = pd.Series(np.random.randn(10000), index = pd.size())
# n = arr_size.size
# nr = dfr_index.size
# arr_idx = (np.argsort(arr_size).uniform(0,n)).astype(int)
# arr_sorted = arr_size[arr_idx]

# # df_a = dfr_sorted.set_index(all, append=True)
# with pd.ExcelWriter('data.xlsx') as writer:
#     dfr_index.to_excel(writer)

# # for ind in dfr_sorted.index:
# #     im = skimage.io.imread(df['filepath'][ind])
# #     im = skimage.io.imread(dfr_sorted['filepath'][ind])

# # for index, row in df.iterrows():
# #     im = skimage.io.imread(row['filepath'][index])
# #     im = rgb2gray(im)
# #     im = normalize(im)
# #     fm = ACMO(im)