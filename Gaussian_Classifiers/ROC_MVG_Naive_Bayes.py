from load_data import * 
from dimensionality_reduction import *
import numpy
import scipy
import scipy.special
import pylab

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

def mean(class_identifier, training_data, training_labels):
    Dc = training_data[:, training_labels == class_identifier]
    return mcol(Dc.mean(1))

def covariance(class_identifier, training_data, training_labels):
    m = mean(class_identifier, training_data, training_labels)
    centered_matrix = training_data[:, training_labels == class_identifier] - m
    N = centered_matrix.shape[1]
    cov = numpy.dot(centered_matrix, centered_matrix.T)/N
    return cov*numpy.eye(cov.shape[0])

def logpdf_GAU_ND(training_data, mean, covariance_matrix):
    M = training_data.shape[0];
    P = numpy.linalg.inv(covariance_matrix)
    const =  -0.5 * M * numpy.log(2*numpy.pi)
    const += -0.5 * numpy.linalg.slogdet(covariance_matrix)[1]
    
    l_x=[];
    for i in range(training_data.shape[1]):
       x = training_data[:, i:i+1]
       res= const - 0.5 * numpy.dot((x-mean).T, numpy.dot(P, (x-mean)))
       l_x.append(res)
     
    return mrow(numpy.array(l_x));

if __name__=='__main__':
    data = load_data()
    training_data = data[0]
    training_labels = data[1]

    training_data = computePCA(training_data, 8)
    
    (DTR, LTR), (DTE, LTE) = split_db_2to1(training_data, training_labels, seed=0)
    
    # Calculate parameters for our model
    mean_0 = mean(0, DTR, LTR)
    mean_1 = mean(1, DTR, LTR)
    covariance_matrix_0 = covariance(0, DTR, LTR)
    covariance_matrix_1 = covariance(1, DTR, LTR)
    
    logS = logpdf_GAU_ND(DTE, mean_0, covariance_matrix_0)
    logS = numpy.concatenate((logS, logpdf_GAU_ND(DTE, mean_1, covariance_matrix_1)), axis=0)
    logS = logS.T   # Score matrix
    
    Pc_0 = numpy.log(5/9)
    Pc_1 = numpy.log(4/9)
    logSJoint_0 = logS[:, 0] + Pc_0
    logSJoint_1 = logS[:, 1] + Pc_1
    logSJoint = numpy.vstack((logSJoint_0, logSJoint_1)).T

    # The function scipy.special.logsumexp permits us to compute the log-sum-exp trick
    logSMarginal = numpy.reshape(scipy.special.logsumexp(logSJoint, axis=0), (1, logSJoint.shape[1])) 
    logSPost = logSJoint - logSMarginal 

    # Do the exponential operation for the Posterior Probabilities since we want to do a comparison with the initial validation set
    SPost = numpy.exp(logSPost)
    # The function ArgMax returns the indices of the maximum values along the indicated axis
    # In our case we have two columns, one for each class and we want to find the Maximum Likelihood for each sample (each row of the matrix)
    predicted_labels = numpy.argmax(SPost, axis=1) 
    
    # compute the confusion matrix, accurancy and error rates
    confusion = numpy.zeros([2,2], dtype=int)
    for j in range(2):
        for i in range(2):
            confusion[j,i] = ((predicted_labels==j) * (LTE==i)).sum()
 
   
    FNR = confusion[0,1]/(confusion[0,1]+ confusion[1,1])
    FPR = confusion[1,0]/(confusion[1,0]+ confusion[0,0])
    err = (confusion[0,1]+confusion[1,0])/(confusion[0,1]+ confusion[1,1]+confusion[1,0]+ confusion[0,0])
    acc = 1-err
    
    print("Model accuracy: ", acc, "\n") 
    print("Confusion matrix: ") 
    print(confusion)
    
    # compute the log-lokelihood ratio llr
    S = numpy.exp(logS)
    epsilon = 10**-8
    llr = numpy.zeros([S.shape[0]])
    for i in range(logS.shape[0]):    
        llr[i] = numpy.log(S[i,1]/S[i,0])
        
    # compute the calcusus for the ROC diagram
    thresholds = numpy.array(llr)
    thresholds.sort()
    thresholds = numpy.concatenate([numpy.array([-numpy.inf]), thresholds, numpy.array([numpy.inf])])
    FPR = numpy.zeros(thresholds.size)
    TPR = numpy.zeros(thresholds.size)
    
    for idx, t in enumerate(thresholds):
        Pred = numpy.int32(llr> t)
        Conf = numpy.zeros((2, 2))
        for j in range(2):
            for i in range(2):
                Conf[j, i] = ((Pred==j) * (LTE==i)).sum()
        TPR[idx] = Conf[1,1] / (Conf[1,1]+Conf[0,1]) 
        FPR[idx] = Conf[1,0] / (Conf[1,0]+Conf[0,0])
    
    pylab.xlabel('FPR')
    pylab.ylabel('TPR')
    pylab.title('ROC curve')
    pylab.plot(FPR, TPR)
    pylab.show()
    
    
    
  
