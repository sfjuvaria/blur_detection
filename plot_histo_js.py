"""
Created on Fri Jan 7 09:10:13 2021
@author: JSyeda
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def plot_hist(x, y, labels, title='', plot_file=''):
  plt.figure()
  plt.bar(x, y, align='center')
  plt.xticks(x, labels)
  plt.title(title)
  if plot_file != '':
    plt.savefig(plot_file)
if __name__ == '__main__':
  # Data
  database_dir = '/home/new/Documents/data_cropped/png/CounterfeitAnalysis/2016-2019CAM_CROP1'
  # data_dir = '/home/new/Documents/data_cropped/png/CounterfeitAnalysis/2016-2019CAM_CROP1/bch'
  # data_dir = '/home/new/Documents/data_cropped/png/CounterfeitAnalysis/2020_test/Android'
  data_dir = '/home/new/Documents/data_cropped/EXP_0729/train'
  # Read CSV
  df = pd.read_csv(os.path.join(database_dir, 'database.csv'))
  dq = pd.read_csv(os.path.join(data_dir, 'image_quality.csv'))
  df['med_path'] = df['med_path'].str.replace(' ','')
  phone_model = []
  for i in range(len(dq.index)):
    dir_name, file_name = os.path.split(dq['image_file'][i])
    file_name = re.sub(r"\s+", "", file_name, flags=re.UNICODE) # remove spaces
    dfm = df[df['med_path'].str.find(file_name)>-1]
    phone_model.append(dfm['Phone_brand'].to_numpy()[0])
  # Add column
  dq['phone_model'] = phone_model
  dq['real_fake'] = (dq['image_file'].str.find('real')>-1) | (dq['image_file'].str.find('Real')>-1)
  # To numpy
  phone_model = np.squeeze(dq[['phone_model']].to_numpy())
  real_fake = np.squeeze(dq[['real_fake']].to_numpy())
  iq = np.squeeze(dq[['iq']].to_numpy())
  # Histogram
  hq = iq>0.15
  st = 'training'
  # st = 'validation'
  # st = 'testing'
  labels, counts = np.unique(phone_model[real_fake==1], return_counts=True)
  ticks = range(len(counts))
  plot_hist(ticks, counts, labels, title=st + '-real-all',
            plot_file=os.path.join(data_dir, 'hist-' + st + '-real-all.png'))
  labels, counts = np.unique(phone_model[real_fake==0], return_counts=True)
  ticks = range(len(counts))
  plot_hist(ticks, counts, labels, title=st + '-fake-all',
            plot_file=os.path.join(data_dir, 'hist-' + st + '-fake-all.png'))
  labels, counts = np.unique(phone_model[np.logical_and(real_fake==1, hq==1)], return_counts=True)
  ticks = range(len(counts))
  plot_hist(ticks, counts, labels, title=st + '-real-hq',
            plot_file=os.path.join(data_dir, 'hist-' + st + '-real-hq.png'))
  labels, counts = np.unique(phone_model[np.logical_and(real_fake==0, hq==1)], return_counts=True)
  ticks = range(len(counts))
  plot_hist(ticks, counts, labels, title=st + '-fake-hq',
            plot_file=os.path.join(data_dir, 'hist-' + st + '-fake-hq.png'))
  labels, counts = np.unique(phone_model[np.logical_and(real_fake==1, hq==0)], return_counts=True)
  ticks = range(len(counts))
  plot_hist(ticks, counts, labels, title=st + '-real-lq',
            plot_file=os.path.join(data_dir, 'hist-' + st + '-real-lq.png'))
  labels, counts = np.unique(phone_model[np.logical_and(real_fake==0, hq==0)], return_counts=True)
  ticks = range(len(counts))
  plot_hist(ticks, counts, labels, title=st + '-fake-lq',
            plot_file=os.path.join(data_dir, 'hist-' + st + '-fake-lq.png'))
  plt.show()
  # Save
  dq.to_csv(os.path.join(data_dir, 'image_quality_model.csv'))
