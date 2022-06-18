# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 15:03:11 2022

@author: genna
"""

import numpy
import scipy.stats
from load_data import *
from Dataset_Analysis import *


def gaussianization(D):
    r= numpy.zeros((D.shape[0], D.shape[1]))
    gauss_D= numpy.zeros((D.shape[0], D.shape[1]))
    
    for index in range(D.shape[0]):
        i=0;
        for feature in D[index,:]:
            r[index, i]= (((D[index, :] < feature).sum()) + 1)/(D.shape[1] + 2)
            gauss_D[index, i]= scipy.stats.norm.ppf(r[index, i])
            i= i + 1
    
    return gauss_D

if __name__=='__main__':
    TD, TL, _, _ = load_data()
    gaussTD= gaussianization(TD)
    plot_hist(gaussTD, TL)
            