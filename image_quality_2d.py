"""
Created on Wed Dec 16 18:02:45 2020
@author: JSyeda
"""
import fnmatch
import os
import re

def findfiles(which, where='.'):
    # '''Returns list of filenames from 'where' path matched by 'which'
    #    shell pattern. Matching is case-insensitive.'''
    # TODO: recursive param with walk() filtering
    rule = re.compile(fnmatch.translate(which), re.IGNORECASE)
    return [name for name in os.listdir(where) if rule.match(name)]
# findfiles('*.jpg')

if __name__ == '__main__':

    # Load data directory folder
    data_dir = '/home/new/Documents/data_imgs'
 
    iff = findfiles( '*.jpg', data_dir)
  