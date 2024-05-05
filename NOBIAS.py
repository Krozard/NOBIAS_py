#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 20:07:05 2023

@author: ziyuanchen
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from saspt import StateArrayDataset, RBME
from Distributions import Defoc_Gaussian
import pyhsmm
from pyhsmm.util.text import progprint_xrange
import pandas as pd
import multiprocessing
from PIL import Image
plt.rcParams["font.family"] = "cursive"
plt.rcParams["figure.dpi"] = 600

class Hyperparameter:
    pass


class parameters:
    pass


class NOBIAS_Dataset(object):
    
    def __init__(self, data, pixel_size_um = 0.160, frame_interval = 0.00748, dz = 0.7 , minLen = 4, \
                 alpha = 1, gamma = 0.1, kappa = 5, Nmax = 10, loc_err = 0.035, fixedstate = False):
        # data is a datafram with datapath and conditions
        
        self.datapath = data["filepath"]
        self.condition = data["condition"]
        self.frame_interval = frame_interval
        self.pixel_size_um = pixel_size_um
        # depth for defoc correction
        self.dz = dz
        self.alpha = alpha
        self.gamma = gamma
        self.kappa = kappa
        self.Nmax = Nmax
        self.minLen = minLen
        self.loc_err = loc_err
        self.fixedstate = fixedstate
        
        
    def _get_tracks(self):
        self.tracks = {}
        self.steps = {}
        for condition in self.condition.unique():
            _cur_trackfiles = self.datapath[self.condition == condition]
            
            # get a list of tracks where each element represent the tracks from that file
            all_tracks = []
            all_steps = []
            for file in _cur_trackfiles:
                dectection = pd.read_csv(file)
                tracks=dectection[["y","x","trajectory","frame"]]
                tracks = tracks.assign(file = file)
                
                grouped = tracks.groupby('trajectory')
    
                track_indi = [grouped.get_group(key) \
                                      for key in grouped.groups if grouped.get_group(key).shape[0]>self.minLen]
                step_indi = []
                for track in track_indi:
                    track_yx = track[["y","x"]].to_numpy()
                    step = track.iloc[:-1]
                    step.loc[:,["y","x"]] = track_yx[1:,:] - track_yx[:-1,:]
                    step_indi.append(step)
                all_tracks = all_tracks + track_indi
                all_steps = all_steps + step_indi
            self.tracks[condition] = pd.concat(all_tracks)
            self.steps[condition] = pd.concat(all_steps)
        return self
    def get_tracks(self):
        if not hasattr(self, 'tracks'):
            self._get_tracks()
        return self.tracks
            
            
    # def _get_steps(self):
    #     self.steps = {}
    #     if not hasattr(self, 'tracks'):
    #         self._get_tracks()
    #     for condition in self.condition.unique():
    #         steps=[]
    #         for track in self.tracks[condition]:
                
                # track_yx = track[["y","x"]].to_numpy()
                # step = track.iloc[:-1]
                # step.loc[:,["y","x"]] = track_yx[1:,:] - track_yx[:-1,:]
                # steps.append(step)
    #         self.steps[condition] = steps 
    #     return self
    def get_steps(self):
        if not hasattr(self, 'steps'):
            # self._get_steps()
            self._get_tracks()
        return self.steps
    def _split_by_traj(self, step_or_track):
           grouped = step_or_track.groupby('trajectory')
           return [grouped.get_group(key) for key in grouped.groups]
       

        

    
    def _build_model(self):
        
        obs_hypparams = {'mu_0':np.zeros(2),
                        'sigma_0':np.eye(2),
                        'kappa_0':0.25,
                        'nu_0':4,
                        'dz': self.dz,
                        'loc_err': self.loc_err}
        if self.fixedstate:
            self.obs_distns = [Defoc_Gaussian(**obs_hypparams) for state in range(self.fixedstate)]
            self.posteriormodel = {}
            for condition in self.condition.unique():
                self.posteriormodel[condition] = pyhsmm.models.HMM(alpha=self.alpha,init_state_concentration=1.,
                    obs_distns=self.obs_distns)
        else:        
            self.obs_distns = [Defoc_Gaussian(**obs_hypparams) for state in range(self.Nmax)]
            self.posteriormodel = {}
            for condition in self.condition.unique():
                self.posteriormodel[condition] = pyhsmm.models.WeakLimitStickyHDPHMM(
                    kappa = self.kappa,alpha=self.alpha,gamma=self.gamma,init_state_concentration=1.,
                    obs_distns=self.obs_distns)
        return self
    def Sample(self, niter = 100):
        if not hasattr(self, 'posteriormodel'):
            self._build_model()
        if not hasattr(self, 'steps'):
            self._get_tracks()
        # key here is condition basically
        for key in self.posteriormodel:
            for steps in self.steps[key]:
                self.posteriormodel[key].add_data(steps[["y","x"]].to_numpy() * self.pixel_size_um)
            for idx in progprint_xrange(niter):
                self.posteriormodel[key].resample_model()
        
        return self
    
    def parallelSample(self, niter = 100):
        if not hasattr(self, 'posteriormodel'):
            self._build_model()
        if not hasattr(self, 'steps'):
            self._get_tracks()
        arg_list =  [(self.posteriormodel[key], split_by_traj_file(self.steps[key], nparray=True), niter, self.pixel_size_um)\
                     for key in self.posteriormodel]
        # Use the map function to parallelize the processing
        # Each process will call the process_item function with an item from the list
        
        temp_posteriormodel_list = parallel_processing(_Indi_Sample,arg_list)
        for key, model in zip(self.steps, temp_posteriormodel_list):
            self.posteriormodel[key] = model
            

        
    def _getD(self ,posteriormodel):
        D=[(posteriormodel.obs_distns[used_state].sigma.trace() - self.loc_err**2)/(4*self.frame_interval) 
           for used_state in posteriormodel.used_states]     
        return D
    def getD_weight(self):
        if not hasattr(self, 'posteriormodel'):
            print('you need to do Sample first to get D value')
            return
        weight = {}
        D = {}
        for key in self.posteriormodel:
            weight[key] = [self.posteriormodel[key].state_usages[used_state] \
                           for used_state in self.posteriormodel[key].used_states]
            D[key] = [(self.posteriormodel[key].obs_distns[used_state].sigma.trace() - self.loc_err**2)
                      /(4*self.frame_interval) 
                      for used_state in self.posteriormodel[key].used_states]  
        return (D, weight)
    
    

def _Indi_Sample(model, steps, niter,pixel_size_um):
    if isinstance(steps[0], pd.DataFrame):
        for step in steps:
            model.add_data(step[["y","x"]].to_numpy() * pixel_size_um)
    else:
        for step in steps:
            model.add_data(step * pixel_size_um)
    # for parallel running the progprint bar messed up
    # for idx in range(niter):
    for idx in progprint_xrange(niter):
        model.resample_model()
    print('a file done', end='\n')
    return model

def parallel_processing(func, input_list):
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes)

    # Use the map function to parallelize the processing
    results = pool.starmap(func, input_list)

    pool.close()
    pool.join()

    return results

def keep_last_n_bases(path, n):
    # Split the path into components
    path = os.path.normpath(path)
    path_list = path.split(os.sep)
    return (os.sep.join(path_list[-n:]))
# multiprocessing run for all tracks with the same type
def replace_values(vector, values_to_replace, replacement_values): # fromChatGPT
    # Create a dictionary to map values to their replacements
    replacement_dict = dict(zip(values_to_replace, replacement_values))

    # Use numpy.vectorize to apply the replacement dictionary to the vector
    vectorized_replace = np.vectorize(lambda x: replacement_dict.get(x, x))

    return vectorized_replace(vector)
    
def reorderSeq(model):
    if not hasattr(model, 'sorted_state_seq'):
        Sigmatrace =[model.obs_distns[used_state].sigma.trace() for used_state in model.used_states]
        NewStateID = np.argsort(Sigmatrace)
        sorted_state_seq = []
        for stateseq in model.stateseqs:
            sorted_state_seq.append(replace_values(stateseq, model.used_states, NewStateID))
        model.sorted_state_seq = sorted_state_seq
        return model
    return
def split_by_traj_file(step_or_track , nparray = False):
    '''
    

    Parameters
    ----------
    step_or_track : pd dataframe
        a large dataframe have trajectory and file column

    Returns
    -------
    list of split individual file and traj/step.

    '''
    step_or_track = step_or_track.sort_values(by=['file', 'trajectory'])
    grouped = step_or_track.groupby(['file', 'trajectory'])
    if nparray:
        return [grouped.get_group(key)[["y","x"]].to_numpy() for key in grouped.groups]
    else:
            
        return [grouped.get_group(key) for key in grouped.groups]
    
# per chatgpt   
def calculate_transition_rates(data):
    transition_counts = {}
    state_counts = {}

    # Iterate over each array in the data
    for sequence in data:
        previous_state = None
        for state in sequence:
            if previous_state is not None:
                # Increment transition count from previous_state to state
                transition_counts[(previous_state, state)] = transition_counts.get((previous_state, state), 0) + 1
            # Increment state count for the current state
            state_counts[state] = state_counts.get(state, 0) + 1
            previous_state = state

    # Calculate transition rates
    transition_rates = {}
    for transition, count in transition_counts.items():
        from_state, to_state = transition
        from_state_count = state_counts[from_state]
        transition_rates[transition] = count / from_state_count

    return transition_rates
    

class NOBIAS_Dataset_allfile(object):
    
    def __init__(self, data, pixel_size_um = 0.160, frame_interval = 0.00748, dz = 0.7 , minLen = 4, \
                 alpha = 1, gamma = 0.1, kappa = 5, Nmax = 10, loc_err = 0.035, fixedstate = False):
        # data is a datafram with datapath and conditions
        
        self.datapath = data["filepath"]
        self.frame_interval = frame_interval
        self.pixel_size_um = pixel_size_um
        # depth for defoc correction
        self.dz = dz
        self.alpha = alpha
        self.gamma = gamma
        self.kappa = kappa
        self.Nmax = Nmax
        self.minLen = minLen
        self.loc_err = loc_err
        self.fixedstate = fixedstate # set to 2/3 to run fixed state HMM
        
        
    def _get_tracks(self):
        self.tracks = []
            # get a list of tracks where each element represent the tracks from that file
        
        for file in self.datapath:
            dectection = pd.read_csv(file)
            _tracks=dectection[["y","x","trajectory","frame"]]
            _tracks = _tracks.assign(file = keep_last_n_bases(file,3))
            
            grouped = _tracks.groupby('trajectory')

            track_indi = [grouped.get_group(key) \
                                  for key in grouped.groups if grouped.get_group(key).shape[0]>self.minLen]
            self.tracks.append(pd.concat(track_indi, ignore_index=True))
        return self
    def get_tracks(self):
        if not hasattr(self, 'tracks'):
            self._get_tracks()
        return self.tracks
            
            
    def _get_steps(self):
        self.steps = []
        if not hasattr(self, 'tracks'):
            self._get_tracks()
    
        for track in self.tracks:
            track_yx = track[["y","x"]].to_numpy()
            trajecory = track[["trajectory"]].to_numpy()
            drop_index,_  = np.nonzero(trajecory[1:]-trajecory[:-1])
            step = track.iloc[:-1]
            step = step.assign(x_pos = step['x'])
            step = step.assign(y_pos = step['y'])
            
            step.loc[:,["y","x"]] = track_yx[1:,:] - track_yx[:-1,:]
            
            step = step[~step.index.isin(drop_index)]
            self.steps.append(step)
        return self
    
    def _split_steps(self,step):
           grouped = step.groupby('trajectory')
           return [grouped.get_group(key) for key in grouped.groups]
    
    def get_steps(self):
        if not hasattr(self, 'steps'):
            self._get_steps()
        return self.steps

    
    def _build_model(self):
        
        obs_hypparams = {'mu_0':np.zeros(2),
                        'sigma_0':np.eye(2),
                        'kappa_0':0.25,
                        'nu_0':4,
                        'dz': self.dz,
                        'loc_err': self.loc_err}
        
        
        # if use fixed state HMM, build a differnt model
        if self.fixedstate:
            self.obs_distns = [Defoc_Gaussian(**obs_hypparams) for state in range(self.fixedstate)]
            self.posteriormodel = [pyhsmm.models.HMM(alpha=self.alpha,init_state_concentration=1.,
                obs_distns=self.obs_distns) for track in self.tracks]
            return self
        
        
        self.obs_distns = [Defoc_Gaussian(**obs_hypparams) for state in range(self.Nmax)]
        self.posteriormodel = [pyhsmm.models.WeakLimitStickyHDPHMM(
            kappa = self.kappa,alpha=self.alpha,gamma=self.gamma,init_state_concentration=1.,
            obs_distns=self.obs_distns) for track in self.tracks]
        
        return self
    

        
    def Sample(self, niter = 100):
        if not hasattr(self, 'steps'):
            self._get_steps()
        
        if not hasattr(self, 'posteriormodel'):
            self._build_model()
        
            
        arg_list =  [(model, self._split_steps(step), niter, self.pixel_size_um) for model, step in zip(self.posteriormodel, self.steps)]
        # Use the map function to parallelize the processing
        # Each process will call the process_item function with an item from the list
        
        self.posteriormodel = parallel_processing(_Indi_Sample,arg_list)

        
    def _getD(self ,posteriormodel):
        D=[(posteriormodel.obs_distns[used_state].sigma.trace() - self.loc_err**2)/(4*self.frame_interval) 
           for used_state in posteriormodel.used_states]     
        return D
    
    def getD_weight(self):
        if not hasattr(self, 'posteriormodel'):
            print('you need to do Sample first to get D value')
            return
        D={}
        weight = {}
        for model, file in zip(self.posteriormodel, self.datapath):
            weight[file] = [model.state_usages[used_state] \
                           for used_state in model.used_states]
            D[file] = [(model.obs_distns[used_state].sigma.trace() - self.loc_err**2)
                      /(4*self.frame_interval) 
                      for used_state in model.used_states]  
        return (D, weight)
    
    def reorderSeq(self):
        for i,model in enumerate(self.posteriormodel):
            Sigmatrace =[model.obs_distns[used_state].sigma.trace() for used_state in model.used_states]
            NewStateID = np.argsort(Sigmatrace)
            sorted_state_seq = []
            for stateseq in model.stateseqs:
                sorted_state_seq.append(replace_values(stateseq, model.used_states, NewStateID))
            self.posteriormodel[i].sorted_state_seq = sorted_state_seq
        return self
    

    
    
class NOBIAS_Dataset_allfileMapping(NOBIAS_Dataset_allfile):
    def __init__(self, data, pixel_size_um = 0.160, frame_interval = 0.00748, dz = 0.7 , minLen = 4, \
                 alpha = 1, gamma = 0.1, kappa = 5, Nmax = 10, loc_err = 0.035, fixedstate = False):
        # data is a datafram with datapath and conditions
        
        super().__init__(data, pixel_size_um, frame_interval, dz, minLen, \
                     alpha, gamma, kappa, Nmax, loc_err, fixedstate)
        self.mapimgpath = data["imagepath"]
        
    def PlotStepState(self, datalabel = ''):
                
        if not hasattr(self, 'posteriormodel'):
            print("Please Sample Posterior Model first")
            return
        if not hasattr(self.posteriormodel[0], 'sorted_state_seq'):
            self.reorderSeq()
        color = ['r', 'g', 'b', 'm', 'c', 'y', 'k']
        for step, model, imagepath in zip(self.steps, self.posteriormodel, self.mapimgpath):
            im = np.array(Image.open(imagepath))
            plt.imshow(im,cmap='gray')
            Allseq = np.concatenate(model.sorted_state_seq)
            for i in np.unique(Allseq):
                index = Allseq == i
                plt.scatter(step.iloc[index]["x_pos"], step.iloc[index]["y_pos"],s=0.5,c=color[i], label = 'state '+str(i+1))
            plt.axis('off')
            plt.legend()
            plt.savefig(imagepath[:-4]+ datalabel +'_stepstate.png')
            plt.close()
        return self
        
    
    
