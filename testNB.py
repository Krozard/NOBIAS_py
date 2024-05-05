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


import glob

from NOBIAS import NOBIAS_Dataset, NOBIAS_Dataset_allfile, reorderSeq, NOBIAS_Dataset_allfileMapping
import multiprocessing

# path = '/home/ziyuanchen/Documents/PAPA/RARRXR/20230502_RARRXR/analysis20230503/sortedTrajectories/NLS/'
# subfolders = [f.path for f in os.scandir(path) if f.is_dir()]




# when there are subfolders
def Getpath(trackfolder):
    
    subfolders = [f.path for f in os.scandir(trackfolder) if f.is_dir()]
    datasets = [os.path.basename(path) for path in subfolders]
    PathDict = {}
    for i, dataset in enumerate(datasets):
        subsubfolders = [f.path for f in os.scandir(subfolders[i]) if f.is_dir()]
        paths_list = []
        for subsubfolder in subsubfolders:
            files = glob.glob(subsubfolder+'/*.csv', recursive=True)
            for file in files:
                # paths_list.append((file, os.path.basename(subfolder))) 
                paths_list.append((file, os.path.basename(file)[:-4])) 
        paths = pd.DataFrame(paths_list, columns=['filepath', 'condition'])
        paths = paths[paths["condition"] != 'other']
        PathDict[dataset] = paths
    return PathDict


# when there are no sub folders


def Run_NOBIAS(paths, niter):
    NOBIASout = NOBIAS_Dataset(paths)
    print('finish dataloading, start sampling')
    NOBIASout.Sample(niter = niter)
    return NOBIASout


def parallel_processing_NOBIAS(PathDict, niter=100):
    num_processes = min(multiprocessing.cpu_count(), len(PathDict))  # Get the number of available CPU cores
    pool = multiprocessing.Pool(processes=num_processes)

    NOBIASout = {}  # Dictionary to store the results

    # Use the map function to parallelize the processing
    # Each process will call the process_item function with an item from the list
    for key, result in zip(PathDict.keys(), \
                           pool.starmap(Run_NOBIAS, zip(PathDict.values(), [niter] * len(PathDict)))):
        NOBIASout[key] = result

    pool.close()
    pool.join()

    return NOBIASout
# datasets = ["WT2"]

def ExtractCellfolder(trackfolder, movedfolder):
    import shutil
    subfolders = [f.path for f in os.scandir(trackfolder) if f.is_dir()]
    for subfolder in subfolders:
        subsubfolders = [f.path for f in os.scandir(subfolder) if f.is_dir()]
        for subsubfolder in subsubfolders:
            new_base_name = os.path.basename(subfolder)+'_'+os.path.basename(subsubfolder)
            if not os.path.isdir(os.path.join(movedfolder, new_base_name)):
                shutil.copytree(subsubfolder, os.path.join(movedfolder, new_base_name))
                
    return

def OrganByCondition(trackfolder):
    subfolders = [f.path for f in os.scandir(trackfolder) if f.is_dir()]
    datasets = [os.path.basename(path) for path in subfolders]
    
    paths_list = []
    for i, dataset in enumerate(datasets):
        subsubfolders = [f.path for f in os.scandir(subfolders[i]) if f.is_dir()]
        
        for subsubfolder in subsubfolders:
            files = glob.glob(subsubfolder+'/*.csv', recursive=True)
            for file in files:
                # paths_list.append((file, os.path.basename(subfolder))) 
                paths_list.append((file, dataset, os.path.basename(file)[:-4])) 
    paths = pd.DataFrame(paths_list, columns=['filepath', 'condition','trackType'])
    paths = paths[paths["trackType"] != 'other']
    
    return paths



if __name__ == '__main__':
    # analysisfolder = '/home/ziyuanchen/Documents/PAPA/RARRXR/20230504_RARRXR/analysis20230522_SA/'
    # trackfolder = analysisfolder+'sortedTrajectories/'
    # # trackfolder = '/home/ziyuanchen/Documents/PAPA/RARRXR/20230504_RARRXR/analysis20230522_SA/sorted_trajectories_cell/'
    # PathDict = Getpath(trackfolder)
    # dataWTPAPA = PathDict["WT"][PathDict["WT"]['condition'] == 'PAPA']
    # dataWTPAPA= dataWTPAPA.sort_values(by='filepath').reset_index(drop=True)
    # maskfolder = '/home/ziyuanchen/Documents/PAPA/RARRXR/20230504_RARRXR/WT' + '/snaps3/'
    # dataWTPAPA["imagepath"] = sorted(glob.glob(maskfolder+'/*.tif', recursive=True))
    # # NOBIASout_WTPAPA = NOBIAS_Dataset_allfileMapping(dataWTPAPA)
    # # NOBIASout_WTPAPA.Sample(niter=100)
    
    # dataWTDR = PathDict["WT"][PathDict["WT"]['condition'] == 'DR']
    # dataWTDR= dataWTDR.sort_values(by='filepath').reset_index(drop=True)
    # maskfolder = '/home/ziyuanchen/Documents/PAPA/RARRXR/20230504_RARRXR/WT' + '/snaps3/'
    # dataWTDR["imagepath"] = sorted(glob.glob(maskfolder+'/*.tif', recursive=True))
    # NOBIASout_WTDR = NOBIAS_Dataset_allfileMapping(dataWTDR)
    # NOBIASout_WTDR.Sample(niter=100)
    
    
    
    imagefolder = '/home/ziyuanchen/Documents/MFM/10_14_2023/ZC002_1dayASV/output/snaps3/'
    analysisfolder = '/home/ziyuanchen/Documents/MFM/10_14_2023/ZC002_1dayASV/output/Analysis_101423/'
    trackfolder = analysisfolder+'sortedTrajectories/'
    PathDict = Getpath(trackfolder)
    # mainfolder = '/home/ziyuanchen/Documents/for_Ziyuan/run2_SCB12_20230822/analysis_20230829/sortedTrajectories/ctrl/exp1/'
    
    PAPAfolder = mainfolder + 'PAPA/'
    DRfolder = mainfolder +'DR/'
    PAPAfiles = sorted(glob.glob(PAPAfolder+'/*.csv', recursive=True))
    DRfiles = sorted(glob.glob(DRfolder+'/*.csv', recursive=True))
    
    PAPAdata=[]
    DRdata=[]
    for file in PAPAfiles:
        PAPAdata.append((file,imagefolder+os.path.basename(file).split('_')[0]+'.tif','PAPA0822'))
    
    # NOBIASout_PAPA = NOBIAS_Dataset_allfileMapping(PAPAdata)
    
    
    for file in DRfiles:
        DRdata.append((file,imagefolder+os.path.basename(file).split('_')[0]+'.tif','DR0822'))
    
    imagefolder0820 = '/home/ziyuanchen/Documents/for_Ziyuan/run1_20230820/snaps3'
    mainfolder0820 = '/home/ziyuanchen/Documents/for_Ziyuan/run1_20230820/analysis20230821/sortedTrajectories/ctrl/exp1/'
    
    PAPAfolder0820 = mainfolder0820 + 'PAPA/'
    DRfolder0820 = mainfolder0820 +'DR/'
    PAPAfiles0820 = sorted(glob.glob(PAPAfolder0820+'/*.csv', recursive=True))
    DRfiles0820 = sorted(glob.glob(DRfolder0820+'/*.csv', recursive=True))
    
    for file in PAPAfiles0820:
        PAPAdata.append((file,imagefolder+os.path.basename(file).split('_')[0]+'.tif','PAPA0820'))
    
    for file in DRfiles0820:
        DRdata.append((file,imagefolder+os.path.basename(file).split('_')[0]+'.tif','DR0820'))
    
    
    
    PAPAdata = pd.DataFrame(PAPAdata, columns=['filepath', 'imagepath', 'condition'])
    DRdata = pd.DataFrame(DRdata, columns=['filepath', 'imagepath', 'condition'])
    # NOBIASout_DR = NOBIAS_Dataset_allfileMapping(DRdata)
    
    NOBIASdata202308 = pd.concat([PAPAdata, DRdata])
    NOBIASout202308 = NOBIAS_Dataset(NOBIASdata202308)
    NOBIASout202308.parallelSample(niter = 100)
    
    # PAPAdata = PathDict["WT"][PathDict["WT"]['condition'] == 'PAPA']
    # NOBIASout_WTPAPA = NOBIAS_Dataset_allfile(PAPAdata)
    # NOBIASout_WTPAPA.Sample(niter=100)
    # DRdata = PathDict["WT"][PathDict["WT"]['condition'] == 'DR']
    # NOBIASout_WTDR = NOBIAS_Dataset_allfile(DRdata)
    # NOBIASout_WTDR.Sample(niter=100)
    
    
    # NOBIASout = parallel_processing_NOBIAS(PathDict)
    # NOBIASout = {}
    # for dataset in PathDict:
    #     NOBIASout[dataset] = NOBIAS_Dataset(PathDict[dataset])
    #     NOBIASout[dataset].Sample(niter = 100)
        
    #     NOBIASout[dataset] = NOBIAS_Dataset(paths, alpha = 10, gamma=0.1, kappa=50)
    #     NOBIASout[dataset].Sample(niter = 100)
    # D={}; weight={}

    # for key in NOBIASout:
    #     D[key], weight[key] = NOBIASout[key].getD_weight()
        
        
    

# dectection = pd.read_csv('/home/ziyuanchen/Documents/PAPA/RARRXR/20230504_RARRXR/analysis20230522_SA/sortedTrajectories/WT/8/PAPA.csv')
# dectection = strobe_multistate(
#     2000,   # 10000 trajectories
#     [0.1, 5],     # diffusion coefficient, microns squared per sec
#     [0.5,0.5],     # state occupancies
#     motion="brownian",
#     geometry="sphere",
#     radius=5.0,
#     dz=0.7,
#     frame_interval=0.01,
#     loc_error=0.035,
#     track_len=100,
#     bleach_prob=0.1
# )

# tracks=dectection[["y","x","trajectory","frame"]]

# grouped = tracks.groupby('trajectory')

# grouped_dataframes = [grouped.get_group(key) for key in grouped.groups if grouped.get_group(key).shape[0]>4]

# def getStep(tracks):
#     steps=[]
#     for track in tracks:
#         track = track.to_numpy()
#         step = track[:-1,:]
#         step[:,[0,1]] = (track[1:,[0,1]] - track[:-1,[0,1]])
#         steps.append(step)
#     return steps
# data = getStep(grouped_dataframes)
        
#########################
#  posterior inference  #
#########################

# Set the weak limit truncation level
# Nmax = 10

# # and some hyperparameters
# obs_dim = 2
# obs_hypparams = {'mu_0':np.zeros(obs_dim),
#                 'sigma_0':np.eye(obs_dim),
#                 'kappa_0':0.25,
#                 'nu_0':obs_dim+2,
#                 'dz': 0.7}
# obs_hypparams = {'mu_0':np.zeros(obs_dim),
#                 'sigma_0':np.eye(obs_dim),
#                 'kappa_0':0.25,
#                 'nu_0':obs_dim+2}
                # 'dz': 0.7}

### HDP-HMM without the sticky bias
# obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in range(Nmax)]
# obs_distns = [Defoc_Gaussian(**obs_hypparams) for state in range(Nmax)]
# posteriormodel = pyhsmm.models.WeakLimitHDPHMM(alpha=6.,gamma=6.,
#                                                init_state_concentration=1.,
#                                    obs_distns=obs_distns)
# for steps in data:
#     posteriormodel.add_data(steps[:,[0,1]])


# for idx in progprint_xrange(100):
#     posteriormodel.resample_model()

# posteriormodel.plot()
# plt.gcf().suptitle('HDP-HMM sampled model after 100 iterations')


# posteriormodel_HMM = posteriormodel

### Sticky-HDP-HMM

# obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in range(Nmax)]
# obs_distns = [Defoc_Gaussian(**obs_hypparams) for state in range(Nmax)]
# posteriormodel = pyhsmm.models.WeakLimitStickyHDPHMM(
#         kappa=5.,alpha=1.,gamma=0.1,init_state_concentration=1.,
#         obs_distns=obs_distns)
# posteriormodel.add_data(data)
# test multiplt dataset
# intervals = np.linspace(0, 2500, num = 101);
# for i in range(100):
#     posteriormodel.add_data(data[np.arange(intervals[i],intervals[i+1]).astype(int),:])

# for steps in data:
#     posteriormodel.add_data(steps[:,[0,1]])
# for idx in progprint_xrange(100):
#     posteriormodel.resample_model()

# posteriormodel.plot()
# plt.gcf().suptitle('Sticky HDP-HMM sampled model after 100 iterations')

# plt.show()
# posteriormodel_stickyHMM_defoc = posteriormodel

# dist = [posteriormodel_stickyHMM.obs_distns[index] for index in posteriormodel_stickyHMM.used_states]