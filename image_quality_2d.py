"""
Created on Wed Dec 16 18:02:45 2020
@author: JSyeda
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io
from skimage.filters import sobel_h, sobel_v, laplace
from skimage.color import rgb2gray, rgb2lab
from skimage.transform import rescale
from scipy.stats import pearsonr, spearmanr
from scipy.ndimage import gaussian_filter

def normalize(x):
  return (x - x.min()) / (x.max() - x.min())

def find_files(which, where='.'):
  file_paths = []
  for dirpath, _, files in os.walk(where):
    for filename in files:
      fname = os.path.join(dirpath,filename)
      if fname.lower().endswith(which):
        file_paths.append(fname)
  return file_paths

def image_info(image_file):
  w = np.zeros((len(image_file),), dtype=int)
  h = np.zeros((len(image_file),), dtype=int)
  s = np.zeros((len(image_file),), dtype=int)
  for i in range(len(image_file)):
    im = io.imread(image_file[i])
    if len(im.shape)==3:
      im = rgb2gray(im)
    h[i], w[i] = im.shape
    s[i] = w[i] * h[i]
  return w, h, s

def image_stats(image_file, w, h):
  lapv = np.zeros((len(image_file),), dtype=float)
  gder = np.zeros((len(image_file),), dtype=float)
  teng = np.zeros((len(image_file),), dtype=float)
  glva = np.zeros((len(image_file),), dtype=float)
  for i in range(len(image_file)):
    im = io.imread(image_file[i])
    if len(im.shape)==3:
      im = rgb2gray(im)
      #imlab = rgb2lab(im)
      #im = imlab[:,:,0]
    scale = h.min()/h[i]
    im = rescale(im, scale)
    # im = normalize(im)
    im = (im - 0.485)/0.229 # from pytorch
    lapv[i] = LAPV(im)
    gder[i] = GDER(im)
    teng[i] = TENG(im)
    glva[i] = GLVA(im)
  return lapv, gder, teng, glva

def plot_scatter(x, y, title='', xlabel='', ylabel='', plot_file=''):
  plt.scatter(x, y)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title)
  if plot_file != '':
    plt.savefig(plot_file)
  plt.show()

def plot_scatter_cmap(x, y, m, title='', xlabel='', ylabel='', plot_file=''):
  cm = plt.get_cmap('jet')
  norm = plt.Normalize(vmin=m.min(), vmax=m.max())
  plt.scatter(x, y, c=m, cmap=cm, norm=norm)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title)
  plt.colorbar()
  if plot_file != '':
    plt.savefig(plot_file)
  plt.show()

def plot_images(p, v, c=3, plot_file=''):
  # Subplots are organized in a Rows x Cols Grid
  # Tot and Cols are known
  t = len(v)
  # Compute Rows required
  r = t // c
  r += t % c
  # Create a Position index
  pos = range(1, t + 1)

  fig = plt.figure()
  fig.subplots_adjust(hspace = 0.5, wspace=.1)
  for i in range(t):
    im = io.imread(p[i])
    ax = fig.add_subplot(r, c, pos[i])
    ax.imshow(im)
    ax.set_title(v[i])
    ax.axis('off')
  plt.tight_layout()
  if plot_file != '':
    plt.savefig(plot_file)
  plt.show()

def LAPV(im):
  # Variance of laplacian (Pech2000)
  fm = laplace(im)
  fm = fm.std()**2
  return fm

def GDER(im, s=3):
  # Gaussian derivative (Geusebroek2000)
  fm = gaussian_filter(im, sigma=s, order=1)
  fm = fm.mean()
  return fm

def TENG(im):
  # Tenengrad (Krotkov86)
  gx = sobel_v(im)
  gy = sobel_h(im)
  fm = gx**2 + gy**2
  fm = fm.mean()
  return fm

def GLVA(im):
  # Graylevel variance (Krotkov86)
  fm = im.std()
  return fm


def data_correlation(a, b):
  # -1.00. A perfect negative (downward sloping) linear relationship
  # -0.70. A strong negative (downward sloping) linear relationship
  # -0.50. A moderate negative (downhill sloping) relationship
  # -0.30. A weak negative (downhill sloping) linear relationship
  #  0.00. No linear relationship
  # +0.30. A weak positive (upward sloping) linear relationship
  # +0.50. A moderate positive (upward sloping) linear relationship
  # +0.70. A strong positive (upward sloping) linear relationship
  # +1.00. A perfect positive (upward sloping) linear relationship

  # Calculate Pearson's correlation
  corr_p, _ = pearsonr(a, b)
  # Calculate spearman's correlation
  corr_s, _ = spearmanr(a, b)
  return corr_p, corr_s

def stats_info(x):
  print('Min: ' + str(x.min()))
  print('Mean: ' + str(x.mean()))
  print('Max: ' + str(x.max()))
  print('Ratio: ' + str(x.max()/x.min()))

def plot_image(im):
  # create figure
  fig = plt.figure(figsize=(10, 10))
  ax = fig.add_subplot(111)

  # add image
  cmap = plt.cm.get_cmap('jet')
  # cmap.set_bad(color='black') # does not work - fix it
  # ax.imshow(im, cmap=cmap, interpolation='none')
  cs = ax.imshow(im, cmap=cmap)
  #ax.imshow(im, cmap=cmap)
  plt.colorbar(cs)

  # hide axis
  plt.axis('off')

  # show
  # plt.show()

def find_string_loc(strings, match):
  idx = -1
  for i, s in enumerate(strings):
    if match in s:
      idx = i
  return idx

if __name__ == '__main__':

  # Image data directory folder
  # data_dir = '/home/new/Documents/data_cropped/png/CounterfeitAnalysis/2016-2019CAM_CROP1/bch'
  # data_dir = '/home/new/Documents/data_cropped/png/CounterfeitAnalysis/2020_test/Android'
  data_dir = '/home/new/Documents/data_cropped/EXP_0729/train'
  
  # Returns list of jpg file paths in a folder
  image_file = find_files('.jpg', data_dir)
  # print(image_file)

  # Image file width, height, and size info
  width, height, size = image_info(image_file)
  # print(size)

  # Image quality measurements
  lapv, gder, teng, glva = image_stats(image_file, width, height)

  # Image quality measure
  iq = (normalize(lapv) + normalize(gder) + normalize(teng) + normalize(glva))/4
  iq = normalize(iq)

  # Convert this list into pandas table for sorting
  df = pd.DataFrame()
  df['image_file'] = image_file
  df['width'] = width
  df['height'] = height
  df['size'] = size
  df['iq'] = iq

  # Sort the list based on iq of the images
  df = df.sort_values(by='iq', ascending= True)

  # Convert table column to numpy vector
  image_file = df[['image_file']].to_numpy()
  width = df[['width']].to_numpy()
  height = df[['height']].to_numpy()
  size = df[['size']].to_numpy()
  iq = df[['iq']].to_numpy()
  image_file = np.squeeze(image_file) # remove singleton dimention
  width = np.squeeze(width)
  height = np.squeeze(height)
  size = np.squeeze(size)
  iq = np.squeeze(iq)

  # Calculate correlation
  corr_p, corr_s = data_correlation(size, iq)
  print('Pearsons correlation: %.3f' % corr_p)
  print('Spearmans correlation: %.3f' % corr_s)
  # plot_scatter(height, iq, 'Height vs IQ', 'Height', 'IQ')
  # plot_scatter_cmap(np.arange(len(image_file)), iq, height, 'Height vs IQ - cmap', 'Images', 'IQ')

  # Plot image quality
  plot_scatter(np.arange(len(image_file)), iq, 'Images vs IQ', 'Images', 'IQ')

  # Plot images with low, mid and high IQ
  nr = 10 # number of random images
  n = len(image_file) # number of all images
  id = np.arange(0, nr)
  plot_images(image_file[id], np.round(iq[id],5))
  id = np.arange(int(n/2)-nr, int(n/2))
  plot_images(image_file[id], np.round(iq[id],5))
  id = np.arange(n-nr, n)
  plot_images(image_file[id], np.round(iq[id],5))

  data_dir_hq = data_dir + '__HQ'
  data_dir_lq = data_dir + '__LQ'
  # create dirs and prepare
  if os.path.exists(data_dir_hq):
      shutil.rmtree(data_dir_hq)
  if os.path.exists(data_dir_lq):
      shutil.rmtree(data_dir_lq)
  os.mkdir(data_dir_lq)
  # copy data
  destination = shutil.copytree(data_dir, data_dir_hq)
  # clean data
  image_file_lq = image_file[iq<0.15]
  for i in range(len(image_file_lq)):
     dir_name, file_name = os.path.split(image_file_lq[i])
     dir_name = dir_name.replace(data_dir, data_dir_hq)
     shutil.move(os.path.join(dir_name, file_name), os.path.join(data_dir_lq, file_name))
  # Save
  df.to_csv(os.path.join(data_dir, 'image_quality.csv'))

# # Find files in folders
#   csv_dir = '/home/new/Documents'
#   csv_file = find_files('.csv', csv_dir)
#   cf = len(csv_file)
#   imgs_num = np.zeros((cf,), dtype=int)
#   # imgs_num = np.array(cf)
  
#   for i in range(len(csv_file)):
#      df_csv = pd.read_csv(csv_file[i]) 
#      imgs_num[i] = len(df_csv.index)