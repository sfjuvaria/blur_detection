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
    plt.show()
    plt.savefig(plot_file)

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
    
    # Add data list + size array into single list
    # datasizes = np.column_stack((data,size))
    df = pd.DataFrame()
    df['image_files']  = image_files
    df['size']  = size
    df['width']  = width
    df['height'] = height
    
    # list(zip(lis,list 2, ...))
    # # Convert this list into pandas table for sorting
    # d = list(zip(image_files, width.tolist(), height.tolist(), size.tolist()))
    # df = pd.DataFrame(d,columns=['filepath', 'width', 'height', 'size'])
    # df = pd.DataFrame([image_files, width, height, size], columns=['filepath', 'width', 'height', 'size'])
    # # df = pd.DataFrame(image_files, datasizes, columns = ['filepath', 'width', 'height', 'size'])
    # Sort the list based on size of the images
    df_sorted = df.sort_values(by='size', ascending= True)
    
    # Convert table to numpy
    data_values = df_sorted.values
    
    # for single column conversion
    size = df_sorted[['size']].to_numpy()
    
    # Plot graph between size and num of images
    num_imgs = np.arange(len(data_values))
    title = 'Scatter plot of image sizes'
    xlabel = 'Image index'
    ylabel = 'Sorted image sizes'
    plot_file = 'scatter1.png'
    scatterplot(num_imgs, size, title, xlabel, ylabel, plot_file)