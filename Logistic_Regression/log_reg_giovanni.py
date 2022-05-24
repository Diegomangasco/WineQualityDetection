# -*- coding: utf-8 -*-
"""
Created on Mon May 16 18:58:05 2022

@author: genna
"""

import numpy
import scipy
import numpy.linalg
import scipy.optimize
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


def logreg_obj_wrap(DTR, LTR, l):
    M= DTR.shape[0]
    Z= 2.0*LTR - 1.0
    def logreg_obj(v):
        w= vcol(v[0:M])
        b= v[-1]       
        S= numpy.dot(w.T, DTR) + b
        # For the non  weighted version of the model
        # cxe= numpy.logaddexp(0, -S*Z) # cross-entropy
        
        # For the prior weighted version of the model
        cxe_one= numpy.logaddexp(0, -S[:, Z>0]*Z[Z>0])
        cxe_minus_one= numpy.logaddexp(0, -S[:, Z<0]*Z[Z<0])
        return 0.5*l*numpy.linalg.norm(w)**2 + (4/9)*cxe_one.mean()+ (5/9)*cxe_minus_one.mean()
    return logreg_obj

    
if __name__=='__main__':
    # D and L are the training data and labels respectively
    D, L, _, _ = load_data()
    D = computePCA(D, 8)
    
    # This commented algorithm is for the split 2 to 1
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
   
    
    for lamb in [0.00001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]:
        confusion_matrix= numpy.zeros([2,2])
        logreg_obj= logreg_obj_wrap(DTR, LTR, lamb)
        _v, _J, _d= scipy.optimize.fmin_l_bfgs_b(logreg_obj, numpy.zeros(DTR.shape[0] +1), approx_grad= True)
        _w= _v[0:DTR.shape[0]]
        _b= _v[-1]
        STE= numpy.dot(_w.T, DTE) + _b
        ZTE= 2.0*LTE - 1.0
        true_negative=0
        true_positive=0
        false_negative=0
        false_positive=0
       
        for i in range(STE.shape[0]):
            if STE[i]<0 and ZTE[i]<0:
                true_negative= true_negative +1
            if STE[i]<0 and ZTE[i]>0:
               false_positive= false_positive +1
            if STE[i]>0 and ZTE[i]>0:
                true_positive=  true_positive +1
            if STE[i]>0 and ZTE[i]<0:
                false_negative= false_negative +1
        
        confusion_matrix[0,0]= true_negative
        confusion_matrix[0,1]= false_negative
        confusion_matrix[1,0]= false_positive
        confusion_matrix[1,1]= true_positive
        
        FNR= false_negative/(false_negative+ true_positive)
        FPR= false_positive/(false_positive+ true_negative)
        
        print(confusion_matrix)
        
        acc= STE[STE*ZTE>0].shape[0]/STE.shape[0]
      
        err= 1- acc
        print(err)
        print("\n")
    
    
    # This is the k-fold algorithm
    # for lamb in [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]:
    
    #     K = 3
    #     K_fold_set = numpy.hsplit(D, K)
    #     K_fold_set = numpy.array(K_fold_set)
    #     K_fold_labels = numpy.split(L, K)
        
    #     accuracy_final_list = []
    #     error_final_list = []
        
    #     for i in range(K):
    #         K_validation_set = K_fold_set[i]
    #         K_validation_label_set = K_fold_labels[i]
    #         # Make a selector and a for cycle for taking the other sections of the training set
    #         selector = [x for x in range(0, K) if x!=i]
    #         K_selected_set = K_fold_set[selector]
    #         K_selected_labels_set = []
    #         for j in range(0, K):
    #             if j != i:
    #                 K_selected_labels_set.append(K_fold_labels[j])
    #         K_training_set = K_selected_set[0]
    #         K_training_labels_set = K_selected_labels_set[0]
    #         # Concatenate the arrays for having one training set both for data and labels
    #         for j in range(1, K-1):
    #             K_training_set = numpy.concatenate((K_training_set, K_selected_set[j]), axis=1)
    #             K_training_labels_set = numpy.concatenate((K_training_labels_set, K_selected_labels_set[j]), axis=0)
            
    #         logreg_obj= logreg_obj_wrap(K_training_set, K_training_labels_set, lamb)
    #         _v, _J, _d= scipy.optimize.fmin_l_bfgs_b(logreg_obj, numpy.zeros(K_training_set.shape[0] +1), approx_grad= True)
    #         _w= _v[0:K_training_set.shape[0]]
    #         _b= _v[-1]
    #         STE= numpy.dot(_w.T, K_validation_set) + _b
    #         ZTE= 2.0*K_validation_label_set - 1.0
    #         acc= STE[STE*ZTE>0].shape[0]/STE.shape[0]
    #         err= 1- acc
    #         accuracy_final_list.append(acc)
    #         error_final_list.append(err)
        
        
    #     accurancy= sum(accuracy_final_list)/K
    #     error= sum(error_final_list)/K
        
    #     print("For lambda: "+str(lamb))
    #     print("accurancy: "+ str(accurancy))
    #     print("error: "+ str(error)+"\n")