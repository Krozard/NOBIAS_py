#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 13:59:05 2023

@author: ziyuanchen
"""

from __future__ import print_function
from builtins import range
from builtins import object
import abc
import numpy as np
import copy

import pybasicbayes
from pybasicbayes.util.stats import combinedata
from pybasicbayes.util.text import progprint_xrange
from future.utils import with_metaclass

from pybasicbayes.distributions import Gaussian
import scipy
from utilities import defoc_corr_sigma

class Defoc_Gaussian(Gaussian):
    #  dz in um
    def __init__(
            self, mu=None, sigma=None,
            mu_0=None, sigma_0=None, kappa_0=None, nu_0=None, dz = 0.7, loc_err = 0.035):
        super().__init__(mu, sigma,mu_0, sigma_0, kappa_0, nu_0)
        self.dz = dz
        self.loc_err = loc_err
    def log_likelihood(self,x):
        try:
            if self.dz is None:
                mu, D = self.mu, self.D
                sigma_chol = self.sigma_chol
                bads = np.isnan(np.atleast_2d(x)).any(axis=1)
                x = np.nan_to_num(x).reshape((-1,D)) - mu
                xs = scipy.linalg.solve_triangular(sigma_chol,x.T,lower=True)
                out = -1./2. * np.einsum('...i,...i', xs.T,xs.T) - D/2*np.log(2*np.pi) \
                    - np.log(sigma_chol.diagonal()).sum()
                out[bads] = 0
                return out
            else:
                mu, D = self.mu, self.D
                sigma_chol = self.sigma_chol
                bads = np.isnan(np.atleast_2d(x)).any(axis=1)
                x = np.nan_to_num(x).reshape((-1,D)) - mu
                xs = scipy.linalg.solve_triangular(sigma_chol,x.T,lower=True)
                out = -1./2. * np.einsum('...i,...i', xs.T,xs.T) - D/2*np.log(2*np.pi) \
                    - np.log(sigma_chol.diagonal()).sum()
                out[bads] = 0
                Dt4 = self.sigma.trace() - self.loc_err**2
                frac_remain = defoc_corr_sigma(Dt4, 1, self.dz)[0]
                
                return out/frac_remain
        except np.linalg.LinAlgError:
            # NOTE: degenerate distribution doesn't have a density
            return np.repeat(-np.inf,x.shape[0])


