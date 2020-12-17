"""
Created on Wed Dec 16 18:02:45 2020
@author: JSyeda
"""
import os
import numpy as np
# import pandas as pd
import skimage
from skimage import io
import matplotlib.pyplot as plt                          

def findfiles(which, where='.'):
  file_paths = []
  for dirpath, dirs, files in os.walk(where):
    for filename in files: 
      fname = os.path.join(dirpath,filename) 
      if fname.lower().endswith(which):
        file_paths.append(fname)
  return file_paths

def image_info(image_files):
    im = skimage.io.imread(image_files)
    w, h, _ = im.shape
    return w,h

def images_info(image_files):
    w = np.zeros((len(image_files), 1), dtype=int)
    h = np.zeros((len(image_files), 1), dtype=int)
    for i in range(len(image_files)):
        im = io.imread(image_files[i])
        # TODO: add if for gray vs RGB
        # if (len(im.shape)==3):
        w[i], h[i], _ = im.shape
            
    return w, h

def scatterplot(num_imgs, sizes, title, xlabel, ylabel,plotname):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.scatter(num_imgs,sizes)
    plt.show()
    plt.savefig(plotname)

if __name__ == '__main__':

    # Load data directory folder
    data_dir = '/home/new/Documents/data_imgs'
    
    # Load data directory with nested folders
    # data_dir = '/home/new/Documents/data_cropped/png/CounterfeitAnalysis/2016-2019CAM_CROP1/bch'
    
    # Returns list of jpg filepaths in a folder 
    image_files = findfiles('.jpg', data_dir)
    print(image_files)
    
    # Call the data loop function to read, width, height info
    width, height = images_info(image_files)
    
    # Calc image size
    size = width * height
    
   