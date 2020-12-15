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

# Headings of the pd table
df = pd.DataFrame(columns=['filepath', 'height', 'width', 'size', 'max', 'min', 'acmo', 'bren', 'lapv'])

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
sorted(img_list)

# Read/open image, convert to grayscale
# Find shape, calc size, find min and max
# Save the values in the row of the pd table
for ind, img_path in enumerate(img_list):
    
    # Change to grayscale
    img = Image.open(img_path).convert('LA')
    im = skimage.io.imread(img_path)
    ims = cv2.imread(img_path)
    # print(img.format)
    # print(img.mode)
    # print(img.size)
    # # show the image
    # img.show()
 # grayscale img
    width, height = img.size
    imax = np.amax(img)
    imin = np.amin(img)
    shp = np.shape(img)
    size = width * height   
    
    imsg = rgb2gray(ims)
    im = rgb2gray(im)
    im = normalize(im)
    acmo = ACMO(im)
    lapv = LAPV(im)
    bren = BREN(im)
    # w,h = im.size
    # iimax = im.max()
    # iimin = im.min()
    # iimmax = iimax.max()
    # iimmin = iimin.min()
    # df = df.append({'filepath':img_path, 'height':height, 'width':width, 'size':size, 'max':imax, 'min':imin }, ignore_index=True)
    # data = {'filepath':img_path, 'height':height, 'width':width, 'size':size, 'max':imax, 'min':imin }
    # df = pd.DataFrame(data)
    df = df.append({'filepath':img_path, 'height':height, 'width':width, 'size':size, 'max':imax, 'min':imin, 'acmo':acmo, 'bren':bren, 'lapv':lapv }, ignore_index=True)

# Sort the list based on size of the images
df_sorted = df.sort_values(by='size', ascending= True)
df_sorted

# Select 100 sample images
dfr = df_sorted.sample(n=100)
# # Sort the list based on size of the images
dfr_sorted = dfr.sort_values(by='size', ascending= True)
dfr_index = dfr_sorted.set_index(['size']).reset_index()

# Plot the graphs-size vs lapv,bren,acmo
dfr_index.plot(x='size', y='lapv') 
dfr_index.plot(x='size', y='bren')
dfr_index.plot(x='size', y='acmo')

# Convert into series format to plot
arr_size = df_sorted['size']
arr_lapv = df_sorted['lapv']

ts = pd.Series(np.random.randn(10000), index = pd.size())
n = arr_size.size
nr = dfr_index.size
arr_idx = (np.argsort(arr_size).uniform(0,n)).astype(int)
arr_sorted = arr_size[arr_idx]

# df_a = dfr_sorted.set_index(all, append=True)
with pd.ExcelWriter('data.xlsx') as writer:
    dfr_index.to_excel(writer)

# for ind in dfr_sorted.index:
#     im = skimage.io.imread(df['filepath'][ind])
#     im = skimage.io.imread(dfr_sorted['filepath'][ind])

# for index, row in df.iterrows():
#     im = skimage.io.imread(row['filepath'][index])
#     im = rgb2gray(im)
#     im = normalize(im)
#     fm = ACMO(im)