"""
Created on Wed Dec 16 18:02:45 2020
@author: JSyeda
"""
import os

def findfiles(which, where='.'):
  file_paths = []
  for dirpath, dirs, files in os.walk(where):
    for filename in files: 
      fname = os.path.join(dirpath,filename) 
      if fname.lower().endswith(which):
        file_paths.append(fname)
  return file_paths


if __name__ == '__main__':

    # Load data directory folder
    # data_dir = '/home/new/Documents/data_imgs'
    
    # Load data directory with nested folders
    data_dir = '/home/new/Documents/data_cropped/png/CounterfeitAnalysis/2016-2019CAM_CROP1/bch'
    
    # Returns list of jpg filepaths in a folder 
    image_files = findfiles('.jpg', data_dir)
    print(image_files)