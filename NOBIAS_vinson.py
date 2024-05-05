#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 18:00:56 2024

@author: ziyuanchen
"""

from __future__ import division
from builtins import range
import numpy as np
np.seterr(divide='ignore') # these warnings are usually harmless for this code
# np.random.seed(0)

from matplotlib import pyplot as plt
import matplotlib
import os
matplotlib.rcParams['font.size'] = 8
import pandas as pd
import pyhsmm
from Distributions import Defoc_Gaussian
from pyhsmm.util.text import progprint_xrange
from PIL import Image
from strobesim import strobe_multistate, FractionalBrownianMotion3D
plt.rcParams["font.family"] = "serif"
plt.rcParams["figure.dpi"] = 600
import pickle

import glob

from NOBIAS import NOBIAS_Dataset, NOBIAS_Dataset_allfile, reorderSeq, NOBIAS_Dataset_allfileMapping
import multiprocessing
if __name__ == '__main__':
        
    trackfolder = '/home/ziyuanchen/Documents/for_vinson'
    
    subfolders = [f.path for f in os.scandir(trackfolder) if f.is_dir()]
    condition_list = [os.path.basename(basefname) for basefname in subfolders]
    paths_list = []
    for basefname, condition in zip(subfolders,condition_list):
        files = glob.glob(basefname+'/*trajs.csv', recursive=True)
        for file in files:
            paths_list.append((file, condition))  
    
    paths = pd.DataFrame(paths_list, columns=['filepath', 'condition'])
    NOBIASout_Vinson_all_3 = NOBIAS_Dataset(paths, Nmax = 3)
    NOBIASout_Vinson_all_3.parallelSample(niter = 200)
    NOBIASout_Vinson_all_4 = NOBIAS_Dataset(paths, Nmax = 4)
    NOBIASout_Vinson_all_4.parallelSample(niter = 200)
    with open('/home/ziyuanchen/Documents/for_vinson/nobias_out.pkl','wb') as fh:
        pickle.dump([NOBIASout_Vinson_all_3, NOBIASout_Vinson_all_4],fh)
    

