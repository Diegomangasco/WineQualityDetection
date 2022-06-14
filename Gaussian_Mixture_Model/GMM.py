import numpy
import scipy.special
import matplotlib
import pylab
from load_data import * 
from dimensionality_reduction import *

def mcol(array):
    return numpy.reshape(array, (array.shape[0], 1))    # Reshape as column array

def mrow(array):
    return numpy.reshape(array, (1, array.shape[0]))    # Reshape as row array

def logpdf_GAU_ND(X, mean, covariance_matrix):
    M = X.shape[0];
    P = numpy.linalg.inv(covariance_matrix)
    print(P)
    const =  -0.5 * M * numpy.log(2*numpy.pi)
    const += -0.5 * numpy.linalg.slogdet(covariance_matrix)[1]
    
    l_x=[];
    for i in range(X.shape[1]):
       x = X[:, i:i+1]
       res= const - 0.5 * numpy.dot((x-mean).T, numpy.dot(P, (x-mean)))
       l_x.append(res)
     
    return mrow(numpy.array(l_x))

def GMM_ll_perSample(X, gmm):
    G = len(gmm)
    N = X.shape[1]
    S = numpy.zeros((G, N))
    for g in range(G):
        S[g, :] = logpdf_GAU_ND(X, gmm[g][1], gmm[g][2]) + numpy.log(gmm[g][0])
    return scipy.special.logsumexp(S, axis=0)

def GMM_EM(X, gmm):
    llNew = None
    llOld = None
    G = len(gmm)
    N = X.shape[1]
    while llOld is None or llNew-llOld > 1e-6:
        llOld=llNew
        SJ = numpy.zeros((G, N))
        for g in range(G):
            SJ[g, :] = logpdf_GAU_ND(X, gmm[g][1], gmm[g][2]) + numpy.log(gmm[g][0])
        SM = scipy.special.logsumexp(SJ, axis=0)
        llNew = SM.sum()/N
        P = numpy.exp(SJ-SM)
        gmmNew = []
        for g in range(G):
            gamma = P[g, :]
            Z = gamma.sum()
            F = (mrow(gamma)*X).sum(1)
            S = numpy.dot(X, (mrow(gamma)*X).T)
            w = Z/N
            mu = mcol(F/Z)
            Sigma = S/Z - numpy.dot(mu, mu.T)
            gmmNew.append((w, mu, Sigma))
        gmm = gmmNew
    return gmm

def conf_matrix(llratio, labs, pr, C_fn, C_fp):
    
    # Computing c* comparing the llr with a threshold t
    t = - numpy.log((pr*C_fn)/((1-pr)*C_fp))
    
    C_star=numpy.zeros([llratio.shape[0],], dtype= int)
    
    for i in range(llratio.shape[0]):
        if llratio[i]>t:
            C_star[i] = 1
        else:
            C_star[i] = 0
            
    # Computing the confusion matrix, comparing labels and c*    
    conf_matr= numpy.zeros([2,2], dtype= int)
    for j in range(2):
        for i in range(2):
            conf_matr[j,i]= ((C_star==j) * (labs==i)).sum()
    return conf_matr

if __name__=='__main__':
    prior_array= [4/9, 1/5, 4/5]
    prior = prior_array[0]
    data = load_data()
    training_data = data[0]
    training_labels = data[1]
    
    training_data = computePCA(training_data, 9)

    K = 5

    real_labels = []
    log_scores = numpy.zeros([1,2])
    
    length_of_interval = int(training_data.shape[1]/K)
    index = 0
    counter = 0
    # For cycle for doing K fold cross validation 
    while index <= (training_data.shape[1] - length_of_interval):
        # Take one section as validation set
        start = index
        if counter < K-1:
            end = index + length_of_interval
        else:
            end = index + length_of_interval + (training_data.shape[1]-K*length_of_interval)
        counter += 1
        K_validation_set = training_data[:, start:end]
        K_validation_label_set = training_labels[start:end]
        K_training_set_part1 = training_data[:, 0:start]
        K_training_set_part2 = training_data[:, end:]
        K_training_set = numpy.concatenate((K_training_set_part1, K_training_set_part2), axis=1)
        K_training_labels_set_part1 = training_labels[0:start]
        K_training_labels_set_part2 = training_labels[end:]
        K_training_labels_set = numpy.concatenate((K_training_labels_set_part1, K_training_labels_set_part2), axis=None)

        index = index + length_of_interval

        gmm = [(0.33, -2*mcol(numpy.ones(9)), 1*numpy.eye(9)), (0.33, 0*mcol(numpy.ones(9)), 1*numpy.eye(9)), (0.33, 2*mcol(numpy.ones(9)), 1*numpy.eye(9))]
        trial_0 = numpy.zeros((9, 1))
        trial_1 = numpy.zeros((9, 1))
        for i in range(K_training_set.shape[1]):
            if(K_training_labels_set[i] == 0):
                trial_0 = numpy.concatenate((trial_0, mcol(K_training_set[:, i])), axis=1)
            else:
                trial_1 = numpy.concatenate((trial_1, mcol(K_training_set[:, i])), axis=1)
        trial_0 = trial_0[:, 1:]
        trial_1 = trial_1[:, 1:]
        gmm_result_0 = GMM_EM(trial_0, gmm)
        gmm_result_1 = GMM_EM(trial_1, gmm)
        print('0: ', gmm_result_0)
        print('1: ', gmm_result_1)


