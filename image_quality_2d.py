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
from scipy.ndimage import laplace       
from skimage.filters import sobel_h, sobel_v     

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
    lapv = LAPV(im)
    # bren = BREN(im)
    return w, h, lapv

def images_info(image_files):
    w = np.zeros((len(image_files),), dtype=int)
    h = np.zeros((len(image_files),), dtype=int)
    lapv = np.zeros((len(image_files),), dtype=float)
    # bren = np.zeros((len(image_files),), dtype=float)
    teng = np.zeros((len(image_files),), dtype=float)
    for i in range(len(image_files)):
        im = io.imread(image_files[i])
        # TODO: add if for gray vs RGB
        # if (len(im.shape)==3):
        w[i], h[i], _ = im.shape
        lapv[i] = LAPV(im)
        # bren[i] = BREN(im)      
        # teng[i] = TENG(im)
    return w, h, lapv

def scatterplot(q, s, title, xlabel, ylabel,plot_file):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.scatter(q, s)
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

def LAPV(im):
  # Variance of laplacian (Pech2000)
  fm = laplace(im)
  fm = fm.std()**2
  return fm

# def BREN(im):
#   # Brenner's (Santos97)
#   (m, n) = im.shape
#   dh = np.zeros((m, n))
#   dv = np.zeros((m, n))
#   dv[0:m-2,:] = im[2:,:] - im[0:-2,:]
#   dh[:,0:n-2] = im[:,2:] - im[:,0:-2]
#   fm = np.maximum(dh, dv)
#   fm = fm**2
#   fm = fm.mean()
#   return fm

def TENG(im):
  # Tenengrad (Krotkov86)
  gx = sobel_v(im)
  gy = sobel_h(im)
  fm = gx**2 + gy**2
  fm = fm.mean()
  return fm
       
if __name__ == '__main__':

    # Load data directory folder
    data_dir = '/home/new/Documents/data_imgs'
    
    # Load data directory with nested folders
    # data_dir = '/home/new/Documents/data_cropped/png/CounterfeitAnalysis/2016-2019CAM_CROP1/acf'
    
    # Returns list of jpg filepaths in a folder 
    image_files = findfiles('.jpg', data_dir)
    print(image_files)
    
    # Call the data loop function to read, width, height info
    width, height, lapvs = images_info(image_files)
    
    # Calc image size
    sizes = width * height
    
    # Convert this list into pandas table for sorting
    df = pd.DataFrame()
    df['image_files']  = image_files
    df['size']  = sizes
    df['width']  = width
    df['height'] = height
    df['LAPV'] = lapvs
    # df['BREN'] = brens
    # df['TENG'] = teng
    
    # Sort the list based on size of the images
    # df_sorted = df.sort_values(by='size', ascending= True)
    df_lapv = df.sort_values(by='LAPV', ascending= True)
    # df_bren = df.sort_values(by='BREN', ascending= True)
    # df_teng = df.sort_values(by='TENG', ascending= True)
    
    # # Convert full table to numpy
    # data_values = df_sorted.values
    
    # # for single column conversion for size
    # image_file = df_sorted[['image_files']].to_numpy()
    # size = df_sorted[['size']].to_numpy()
    # lapv = df_sorted[['LAPV']].to_numpy()
    # # To remove brackets
    # image_file = np.squeeze(image_file)
    # size = np.squeeze(size)
    # lapv = np.squeeze(lapv)
    
    # for single column conversion for lapv
    image_file = df_lapv[['image_files']].to_numpy()
    size = df_lapv[['size']].to_numpy()
    lapv = df_lapv[['LAPV']].to_numpy()
    # To remove brackets
    image_file = np.squeeze(image_file)
    size = np.squeeze(size)
    lapv = np.squeeze(lapv)
    
    # # for single column conversion for bren
    # image_file = df_bren[['image_files']].to_numpy()
    # bren = df_bren[['BREN']].to_numpy()
    # # To remove brackets
    # image_file = np.squeeze(image_file)
    # bren = np.squeeze(bren)
    
    # # for single column conversion for teng
    # image_file = df_teng[['image_files']].to_numpy()
    # teng = df_teng[['TENG']].to_numpy()
    # # To remove brackets
    # image_file = np.squeeze(image_file)
    # teng = np.squeeze(teng)
    
    # # Plot graph between size and num of images
    # num_imgs = np.arange(len(image_file))
    # title = 'Scatter plot of image lapv'
    # xlabel = 'Lapv'
    # ylabel = 'Image size'
    # plot_file = 'scatter1_lapv.png'
    # scatterplot(lapv, size, title, xlabel, ylabel, plot_file)
    
    fig = plt.figure() 
    n = len(image_files)  #  10000 # number of all images
    nr = 9 # number of random images 
    id = (np.random.uniform(0,n,nr)).astype(int) # image index
    id = np.unique(id) # make sure that they are unique id'
    random_files = image_file[id]
    # random_sizes = size[id]
    random_lapvs = lapv[id]
    plot_images(random_files, random_lapvs)
    
