# -*- coding: utf-8 -*-
"""
Created on Tue May 31 15:31:52 2022

@author: genna
"""

import numpy
import numpy
import matplotlib
import matplotlib.pyplot as plt
from load_data import *
from dimensionality_reduction import *

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)


def vcol(v):
    return v.reshape((v.size, 1))

def vrow(v):
    return v.reshape((1, v.size))


if __name__=='__main__':
    data = load_data()
    training_data = data[0]
    training_labels = data[1]
    (DTR, LTR), (DTE, LTE)= split_db_2to1(training_data, training_labels)
    _, U= computeLDA(DTR,  LTR, 1)
    DP= numpy.dot(U.T, DTE)
    thresholds = DP.ravel()
    thresholds.sort()
    thresholds = numpy.concatenate([numpy.array([-numpy.inf]), thresholds, numpy.array([numpy.inf])])
    errs= numpy.zeros(thresholds.size)
    for idx, t in enumerate(thresholds):
        Pred = numpy.int32(DP < t)
        Conf = numpy.zeros((2, 2))
        for j in range(2):
            for i in range(2):
                # Confusion matrix for each threshold
                Conf[j, i] = ((Pred==j) * (LTE==i)).sum()
        errs[idx] = (Conf[0,1]+Conf[1,0])/(Conf[0,1]+Conf[1,1]+Conf[1,0]+Conf[0,0])
    
    err_min= min(errs)      
    # print("Best Threshold: ", best_t)
    print("Error: ", err_min)
    print("Accurancy: ", (1-err_min))
    
    # --------------------------------------
    
    # K = 5
    # real_labels = []  
    # DP_lda= []
    # length_of_interval = int(training_data.shape[1]/K)
    # index = 0
    # counter = 0
    # # while cycle for doing K fold cross validation 
    # while index <= (training_data.shape[1] - length_of_interval):
    #     # Take one section as validation set
    #     start = index
    #     if counter < K-1:
    #         end = index + length_of_interval
    #     else:
    #         end = index + length_of_interval + (training_data.shape[1]-K*length_of_interval)
    #     counter += 1
    #     K_validation_set = training_data[:, start:end]
    #     K_validation_label_set = training_labels[start:end]
    #     K_training_set_part1 = training_data[:, 0:start]
    #     K_training_set_part2 = training_data[:, end:]
    #     K_training_set = numpy.concatenate((K_training_set_part1, K_training_set_part2), axis=1)
    #     K_training_labels_set_part1 = training_labels[0:start]
    #     K_training_labels_set_part2 = training_labels[end:]
    #     K_training_labels_set = numpy.concatenate((K_training_labels_set_part1, K_training_labels_set_part2), axis=None)

    #     index = index + length_of_interval
    
    #     real_labels = numpy.concatenate((real_labels, K_validation_label_set), axis=0)
    #     _, U= computeLDA(K_training_set,  K_training_labels_set, 1)
    #     DP= numpy.dot(U.T, K_validation_set)
    #     DP_lda= numpy.concatenate((DP_lda, DP.ravel()), axis=0)
        
    
        
   
    # thresholds = DP_lda
    # thresholds.sort()
    # thresholds = numpy.concatenate([numpy.array([-numpy.inf]), thresholds, numpy.array([numpy.inf])])
    # errs= numpy.zeros(thresholds.size)
    # for idx, t in enumerate(thresholds):
    #     Pred = numpy.int32(DP_lda < t)
    #     Conf = numpy.zeros((2, 2))
    #     for j in range(2):
    #         for i in range(2):
    #             # Confusion matrix for each threshold
    #             Conf[j, i] = ((Pred==j) * (real_labels==i)).sum()
    #     errs[idx] = (Conf[0,1]+Conf[1,0])/(Conf[0,1]+Conf[1,1]+Conf[1,0]+Conf[0,0])
    
    
    # err_min= min(errs)      
    # # print("Best Threshold: ", best_t)
    # print("Error: ", err_min)
    # print("Accurancy: ", (1-err_min))
            
    
    