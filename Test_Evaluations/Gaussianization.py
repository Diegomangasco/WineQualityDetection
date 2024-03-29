import numpy
import scipy.stats


def gaussianization(D_train, D_test):
    r= numpy.zeros((D_test.shape[0], D_test.shape[1]))
    gauss_D_test= numpy.zeros((D_test.shape[0], D_test.shape[1]))
    
    for index in range(D_test.shape[0]):
        i=0;
        for element in D_test[index,:]:
            r[index, i]= (((D_train[index, :] < element).sum()) + 1)/(D_train.shape[1] + 2)
            gauss_D_test[index, i]= scipy.stats.norm.ppf(r[index, i])
            i= i + 1

    r= numpy.zeros((D_train.shape[0], D_train.shape[1]))
    gauss_D_train= numpy.zeros((D_train.shape[0], D_train.shape[1]))
    
    for index in range(D_train.shape[0]):
        i=0;
        for element in D_train[index,:]:
            r[index, i]= (((D_train[index, :] < element).sum()) + 1)/(D_train.shape[1] + 2)
            gauss_D_train[index, i]= scipy.stats.norm.ppf(r[index, i])
            i= i + 1
    
    return gauss_D_train, gauss_D_test

     
