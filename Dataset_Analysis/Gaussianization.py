import numpy
import scipy.stats
from load_data import *
from Dataset_Analysis import *


def gaussianization(D):
    r= numpy.zeros((D.shape[0], D.shape[1]))
    # r is the rank matrix
    gauss_D= numpy.zeros((D.shape[0], D.shape[1]))
    
    for index in range(D.shape[0]):
        i=0;
        for element in D[index,:]:
            r[index, i]= (((D[index, :] < element).sum()) + 1)/(D.shape[1] + 2)
            gauss_D[index, i]= scipy.stats.norm.ppf(r[index, i])
            # The scipy.stats.norm.ppf returns the inverse of the cumulative distribution function of the standard normal distribution
            i= i + 1
    
    return gauss_D

if __name__=='__main__':
    TD, TL, _, _ = load_data()
    gaussTD= gaussianization(TD)
    plot_hist(gaussTD, TL)
            