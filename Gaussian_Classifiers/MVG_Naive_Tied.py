from load_data import * 
from dimensionality_reduction import *
from Gaussianization import *
import numpy
import scipy
import scipy.special
import pylab
import matplotlib

def class_columns(class_identifier, training_data, training_labels):
    return training_data[:, training_labels == class_identifier].shape[1]

def mean(class_identifier, training_data, training_labels):
    Dc = training_data[:, training_labels == class_identifier]
    return mcol(Dc.mean(1))

def covariance(class_identifier, training_data, training_labels):
    m = mean(class_identifier, training_data, training_labels)
    centered_matrix = training_data[:, training_labels == class_identifier] - m
    N = centered_matrix.shape[1]
    return numpy.dot(centered_matrix, centered_matrix.T)/N

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
     
    return mrow(numpy.array(l_x))


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
    prior = prior_array[2]
    data = load_data()
    training_data = data[0]
    training_labels = data[1]
    
    #training_data = gaussianization(training_data)
    
    training_data = computePCA(training_data, 7)

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
        
        # Train the model
        mean_0 = mean(0, K_training_set, K_training_labels_set)
        mean_1 = mean(1, K_training_set, K_training_labels_set)
        covariance_matrix_0 = covariance(0, K_training_set, K_training_labels_set)
        covariance_matrix_1 = covariance(1, K_training_set, K_training_labels_set)

        N_0 = class_columns(0, K_training_set, K_training_labels_set)
        N_1 = class_columns(1, K_training_set, K_training_labels_set)
        tied_covariance = (1/K_training_set.shape[1])*(covariance_matrix_0*N_0 + covariance_matrix_1*N_1)
        tied_covariance = tied_covariance*numpy.eye(tied_covariance.shape[0])

        # Calculate the likelihood for the validation set
        logS = logpdf_GAU_ND(K_validation_set, mean_0, tied_covariance)
        logS = numpy.concatenate((logS, logpdf_GAU_ND(K_validation_set, mean_1, tied_covariance)), axis=0)
        logS = logS.T
        log_scores = numpy.concatenate((log_scores, logS))
        real_labels = numpy.concatenate((real_labels, K_validation_label_set), axis=0)

    log_scores = log_scores[1:, :]
    # We assume that the prior probability of each class is 1/2
    Pc_0 = numpy.log(1-prior)
    Pc_1 = numpy.log(prior)
    logSJoint_0 = log_scores[:, 0] + Pc_0
    logSJoint_1 = log_scores[:, 1] + Pc_1
    logSJoint = numpy.vstack((logSJoint_0, logSJoint_1)).T

    # The function scipy.special.logsumexp permits us to compute the log-sum-exp trick
    logSMarginal = numpy.reshape(scipy.special.logsumexp(logSJoint, axis=0), (1, logSJoint.shape[1])) 
    logSPost = logSJoint - logSMarginal 

    # Do the exponential operation for the Posterior Probabilities since we want to do a comparison with the initial validation set
    SPost = numpy.exp(logSPost)
    # The function ArgMax returns the indices of the maximum values along the indicated axis
    # In our case we have two columns, one for each class and we want to find the Maximum Likelihood for each sample (each row of the matrix)
    Predicted_labels = numpy.argmax(SPost, axis=1) 

    # Compute the confusion matrix, accurancy and error rates
    confusion = numpy.zeros([2,2], dtype=int)
    for j in range(2):
        for i in range(2):
            confusion[j,i] = ((Predicted_labels==j) * (real_labels==i)).sum()
 
    FNR_ = confusion[0,1]/(confusion[0,1]+ confusion[1,1])
    FPR_ = confusion[1,0]/(confusion[1,0]+ confusion[0,0])
    err = (confusion[0,1]+confusion[1,0])/(confusion[0,1]+confusion[1,1]+confusion[1,0]+confusion[0,0])
    acc = 1-err
    print("Model accuracy:", round(acc*100, 3), "%")
    print("Model error:", round(err*100, 3), "%")
    print("Confusion matrix: ") 
    print(confusion)

    # Compute the log-lokelihood ratio llr by using the score matrix
    S = numpy.exp(log_scores)
    llr = numpy.zeros([S.shape[0]])
    for i in range(log_scores.shape[0]):    
        llr[i] = numpy.log(S[i,1]/S[i,0])
        
    # Compute the calcusus for the ROC diagram
    thresholds = numpy.array(llr)
    thresholds.sort()
    thresholds = numpy.concatenate([numpy.array([-numpy.inf]), thresholds, numpy.array([numpy.inf])])
    FPR = numpy.zeros(thresholds.size)
    TPR = numpy.zeros(thresholds.size)
    
    for idx, t in enumerate(thresholds):
        Pred = numpy.int32(llr > t)
        Conf = numpy.zeros((2, 2))
        for j in range(2):
            for i in range(2):
                # Confusion matrix for each threshold
                Conf[j, i] = ((Pred==j) * (real_labels==i)).sum()
        TPR[idx] = Conf[1,1] / (Conf[1,1]+Conf[0,1]) 
        FPR[idx] = Conf[1,0] / (Conf[1,0]+Conf[0,0])
    
    pylab.xlabel('FPR')
    pylab.ylabel('TPR')
    pylab.title('ROC curve')
    pylab.plot(FPR, TPR)
    pylab.show()

    # Compute the normalized DCF of our model with a threshold that is our prior for computing the rates
    Cost_matrix = numpy.zeros([2,2], dtype= int)
    Cfn = 1
    Cfp = 1
    Cost_matrix[0,1] = Cfn
    Cost_matrix[1,0] = Cfp
    # Compute the confusion matrix with the llr calculated previously and with real_labels from the k fold 
    confusion__matrix = conf_matrix(llr, real_labels, prior, Cfn, Cfp)
    FNR_DCF = confusion__matrix[0,1] / ( confusion__matrix[0,1] + confusion__matrix[1,1])
    FPR_DCF = confusion__matrix[1,0] / ( confusion__matrix[1,0] + confusion__matrix[0,0])
    # Bayes empirical risk
    Bemp = prior*Cfn*FNR_DCF + (1-prior)*Cfp*FPR_DCF
    # Bayes empirical risk with a dummy strategy
    Bdummy = min(prior*Cfn, (1-prior)*Cfp) 
    # Normalized DCF
    normDCF = Bemp/Bdummy
    print("Actual normalized DCF", round(normDCF, 3))
    
 
    # Compute the minimum normalized DCF for our model
    Bempirical = numpy.zeros(thresholds.size)
    for idx, t in enumerate(thresholds):
        Pred = numpy.int32(llr > t)
        Conf = numpy.zeros((2, 2))
        for j in range(2):
            for i in range(2):
                Conf[j, i] = ((Pred==j) * (real_labels==i)).sum()
        FNR_minDCF = Conf[0,1] / (Conf[0,1] + Conf[1,1])
        FPR_minDCF = Conf[1,0] / (Conf[1,0] + Conf[0,0])
        Bempirical[idx]= prior*Cfn*FNR_minDCF + (1-prior)*Cfp*FPR_minDCF
   
    Bemp_min= Bempirical.min()   
    min_normDCF= Bemp_min/Bdummy
    print("Minimum normalized DCF:", round(min_normDCF, 3))
    
    
    # Compute the Bayes error plot for our recognizer. Consider values of p˜ ranging, for example, from -3
    # to +3. You can generate linearly spaced values (21 is the number of points we evaluate the DCF at in the example, i.e. the number of values of p˜
    # we consider). For each value p˜, compute the corresponding effective prior
    
    effPriorLogOdds = numpy.linspace(-3, 3, 21) 
    normalizedDCF = numpy.zeros(21)
    minDCF = numpy.zeros(21)
    
    for  index in range(21):
        eff_prior= 1/ (1 + numpy.exp( - effPriorLogOdds[index]))
        # Compute the actual DCF
        Bdummy = min(eff_prior*Cfn, (1-eff_prior)*Cfp)
        c_m = conf_matrix(llr, real_labels, eff_prior, Cfn, Cfp)
        FNR_actualDCF =  c_m[0,1] / (c_m[0,1] + c_m[1,1])
        FPR_actualDCF =  c_m[1,0] / (c_m[1,0] + c_m[0,0])
        Bemp_actualDCF = eff_prior*Cfn*FNR_actualDCF + (1-eff_prior)*Cfp*FPR_actualDCF
        normBemp= Bemp_actualDCF/Bdummy
        normalizedDCF[index]= normBemp
        
        # Compute the min DCF
        Bemp_minDCF= numpy.zeros(thresholds.size)
        for idx, t in enumerate(thresholds):
            Pred = numpy.int32(llr > t)
            Conf = numpy.zeros((2, 2))
            for j in range(2):
                for i in range(2):
                    Conf[j, i]= ((Pred==j) * (real_labels==i)).sum()
            FNR_minDCF = Conf[0,1] / (Conf[0,1] + Conf[1,1])
            FPR_minDCF = Conf[1,0] / (Conf[1,0] + Conf[0,0])
            Bemp_minDCF[idx]= eff_prior*Cfn*FNR_minDCF + (1-eff_prior)*Cfp*FPR_minDCF
             
        Bemp_min= Bemp_minDCF.min()   
        
        minDCF[index]= Bemp_min/Bdummy
        
        
    matplotlib.pyplot.xlabel('log(π/(1-π))')
    matplotlib.pyplot.ylabel('DCF')
    matplotlib.pyplot.title('Bayes error plot')
    matplotlib.pyplot.plot(effPriorLogOdds, normalizedDCF, color='r',  label= 'DCF')
    matplotlib.pyplot.plot(effPriorLogOdds, minDCF, color='b', label= 'min DCF')
    matplotlib.pyplot.ylim([0, 1.1])
    matplotlib.pyplot.xlim([-3, 3])
    matplotlib.pyplot.legend()
    matplotlib.pyplot.show()



