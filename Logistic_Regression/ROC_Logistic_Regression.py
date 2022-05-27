import numpy
import scipy
import numpy.linalg
import scipy.optimize
import pylab
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
    M = DTR.shape[0]
    Z = 2.0*LTR - 1.0
    def logreg_obj(v):
        w = vcol(v[0:M])
        b = v[-1]       
        S = numpy.dot(w.T, DTR) + b
        # For the non  weighted version of the model
        # cxe = numpy.logaddexp(0, -S*Z) # cross-entropy
        # For the prior weighted version of the model
        cxe_one = numpy.logaddexp(0, -S[:, Z>0]*Z[Z>0])
        cxe_minus_one = numpy.logaddexp(0, -S[:, Z<0]*Z[Z<0])
        return 0.5*l*numpy.linalg.norm(w)**2 + (4/9)*cxe_one.mean()+ (5/9)*cxe_minus_one.mean()
    return logreg_obj

    
if __name__=='__main__':
    # D and L are the training data and labels respectively
    D, L, _, _ = load_data()
    D = computePCA(D, 8)
    
    # This commented algorithm is for the split 2 to 1
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
   
    for lamb in [0.00001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]:
        confusion_matrix = numpy.zeros([2,2])
        logreg_obj = logreg_obj_wrap(DTR, LTR, lamb)
        _v, _J, _d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, numpy.zeros(DTR.shape[0] + 1), approx_grad= True)
        _w = _v[0:DTR.shape[0]]
        _b = _v[-1]
        STE = numpy.dot(_w.T, DTE) + _b
        ZTE = 2.0*LTE - 1.0
        true_negative = 0
        true_positive = 0
        false_negative = 0
        false_positive = 0
       
        for i in range(STE.shape[0]):
            if STE[i]<0 and ZTE[i]<0:
                true_negative = true_negative +1
            if STE[i]<0 and ZTE[i]>0:
               false_positive = false_positive +1
            if STE[i]>0 and ZTE[i]>0:
                true_positive =  true_positive +1
            if STE[i]>0 and ZTE[i]<0:
                false_negative = false_negative +1
        
        confusion_matrix[0,0] = true_negative
        confusion_matrix[0,1] = false_negative
        confusion_matrix[1,0] = false_positive
        confusion_matrix[1,1] = true_positive
        
        FNR = false_negative/(false_negative + true_positive)
        FPR = false_positive/(false_positive + true_negative)
        
        llr = STE - numpy.log((4/9)/(1-4/9))
        thresholds = llr
        thresholds.sort()

        thresholds = numpy.concatenate([numpy.array([-numpy.inf]), thresholds, numpy.array([numpy.inf])])
        FPR = numpy.zeros(thresholds.size)
        TPR = numpy.zeros(thresholds.size)
    
        for idx, th in enumerate(thresholds):
            Pred = numpy.int32(llr > th)
            Conf = numpy.zeros((2, 2))
            for j in range(0, 2):
                for i in range(0, 2):
                    Conf[j, i]= ((Pred==j) * (LTE==i)).sum()
            TPR[idx]= Conf[1,1] / (Conf[1,1]+Conf[0,1]) 
            FPR[idx]= Conf[1,0] / (Conf[1,0]+Conf[0,0])
       
        acc = STE[STE*ZTE>0].shape[0]/STE.shape[0]
        err = 1 - acc
        print("Lambda value:", lamb)
        print("Model accuracy:", acc)
        print("Model error:", err)
        print("Confusion matrix:")
        print(confusion_matrix)
        print("\n")
        pylab.xlabel('FPR')
        pylab.ylabel('TPR')
        pylab.title('ROC curve')
        pylab.plot(FPR, TPR)
        pylab.show()