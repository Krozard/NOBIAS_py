#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 20:07:25 2023

@author: ziyuanchen
"""

import numpy as np

def defoc_corr_sigma(Dt4, n_frames, dz):
    """
    Adapted from saspt package f_remain_rbm by Alec Heckert
    Calculate the fraction of regular Brownian particles that 
    remain in a microscope's depth of field after some number 
    of frames.
    args
    ----
        Dt4             :   trace of Guassian Sigma - locerr^2
        n_frames        :   int, the number of frames
        dz              :   float, depth of field in um
    returns
    -------
        1D ndarray of shape (n_frames,), the probability
            to remain at each frame interval
    """

    # Support for the calculations
    s = (int(dz//2.0)+1) * 2
    support = np.linspace(-s, s, int(((2*s)//0.001)+2))[:-1]
    hz = 0.5 * dz 
    inside = np.abs(support) <= hz 
    outside = ~inside 

    # Define the transfer function for this BM
    g = np.exp(-(support ** 2)/Dt4)
    g /= g.sum()
    g_rft = np.fft.rfft(g)   

    # Set up the initial probability density
    pmf = inside.astype(np.float64)
    pmf /= pmf.sum()

    # Propagate over subsequent frame intervals
    result = np.zeros(n_frames, dtype=np.float64)
    for t in range(n_frames):
        pmf = np.fft.fftshift(np.fft.irfft(
            np.fft.rfft(pmf) * g_rft, n=pmf.shape[0]))
        pmf[outside] = 0.0
        result[t] = pmf.sum()

    return result 
