"""
Created on Wed Dec 16 18:02:45 2020
@author: JSyeda
"""
import os
import numpy as np
import pandas as pd
import skimage
from skimage import io
import matplotlib.pyplot as plt              
# import random            

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
    w = np.zeros((len(image_files),), dtype=int)
    h = np.zeros((len(image_files),), dtype=int)
    for i in range(len(image_files)):
        im = io.imread(image_files[i])
        # TODO: add if for gray vs RGB
        # if (len(im.shape)==3):
        w[i], h[i], _ = im.shape
            
    return w, h

def scatterplot(num_imgs, size, title, xlabel, ylabel,plot_file):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.scatter(num_imgs,size)
    # plt.grid(True)
    plt.show()
    plt.savefig(plot_file)

def plot_images(p,v):
    fig, axs = plt.subplots(1,nr, figsize=(25,5))
    fig.subplots_adjust(hspace = 0.5, wspace=.1)  
    for i in range(len(p)):
        im = io.imread(p[i])
        axs[i].imshow(im)
        axs[i].set_title(v[i])      
        axs[i].axis('off')           
    # plt.axes("off")   
    plt.tight_layout()
    plt.show()
    plt.savefig('img_plots.png')
        
if __name__ == '__main__':

    # Load data directory folder
    # data_dir = '/home/new/Documents/data_imgs'
    
    # Load data directory with nested folders
    data_dir = '/home/new/Documents/data_cropped/png/CounterfeitAnalysis/2016-2019CAM_CROP1/acf'
    
    # Returns list of jpg filepaths in a folder 
    image_files = findfiles('.jpg', data_dir)
    print(image_files)
    
    # Call the data loop function to read, width, height info
    width, height = images_info(image_files)
    
    # Calc image size
    size = width * height
    
    # Convert this list into pandas table for sorting
    df = pd.DataFrame()
    df['image_files']  = image_files
    df['size']  = size
    df['width']  = width
    df['height'] = height
    
    # Sort the list based on size of the images
    df_sorted = df.sort_values(by='size', ascending= True)
    
    # # Convert full table to numpy
    # data_values = df_sorted.values
    
    # for single column conversion
    size = df_sorted[['size']].to_numpy()
    image_file = df_sorted[['image_files']].to_numpy()
    
    size = np.squeeze(size)
    image_file = np.squeeze(image_file)
    
    # Plot graph between size and num of images
    num_imgs = np.arange(len(size))
    title = 'Scatter plot of image sizes'
    xlabel = 'Image index'
    ylabel = 'Sorted image sizes'
    plot_file = 'scatter1.png'
    scatterplot(num_imgs, size, title, xlabel, ylabel, plot_file)
    
    fig = plt.figure() 
    n = len(image_files)  #  10000 # number of all images
    nr = 9 # number of random images 
    id = (np.random.uniform(0,n,nr)).astype(int) # image index
    id = np.unique(id) # make sure that they are unique id'
    random_files = image_file[id]
    random_sizes = size[id]
    plot_images(random_files, random_sizes)
    
