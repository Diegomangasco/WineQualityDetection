from load_data import * 
from dimensionality_reduction import *
import numpy
import scipy
import scipy.special

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

def MVG():
    data = load_data()
    training_data = data[0]
    training_labels = data[1]

    # training_data = computePCA(training_data, 10)
    # training_data = computeLDA(training_data, training_labels, 7)

    K = 3
    K_fold_set = numpy.hsplit(training_data, K)
    K_fold_set = numpy.array(K_fold_set)
    K_fold_labels = numpy.split(training_labels, K)

    accuracy_final_list = []
    error_final_list = []
    
    # For cycle for doing K fold cross validation 
    for i in range(0, K):
        # Take one section as validation set
        K_validation_set = K_fold_set[i]
        K_validation_label_set = K_fold_labels[i]
        # Make a selector and a for cycle for taking the other sections of the training set
        selector = [x for x in range(0, K) if x!=i]
        K_selected_set = K_fold_set[selector]
        K_selected_labels_set = []
        for j in range(0, K):
            if j != i:
                K_selected_labels_set.append(K_fold_labels[j])
        K_training_set = K_selected_set[0]
        K_training_labels_set = K_selected_labels_set[0]
        # Concatenate the arrays for having one training set both for data and labels
        for j in range(1, K-1):
            K_training_set = numpy.concatenate((K_training_set, K_selected_set[j]), axis=1)
            K_training_labels_set = numpy.concatenate((K_training_labels_set, K_selected_labels_set[j]), axis=0)
        
        # Train the model
        mean_0 = mean(0, K_training_set, K_training_labels_set)
        mean_1 = mean(1, K_training_set, K_training_labels_set)
        covariance_matrix_0 = covariance(0, K_training_set, K_training_labels_set)
        covariance_matrix_1 = covariance(1, K_training_set, K_training_labels_set)

        # Calculate the likelihood for the validation set
        logS = logpdf_GAU_ND(K_validation_set, mean_0, covariance_matrix_0)
        logS = numpy.concatenate((logS, logpdf_GAU_ND(K_validation_set, mean_1, covariance_matrix_1)), axis=0)
        logS = logS.T
        
        # We assume that the prior probability of each class is 1/2
        Pc_0 = numpy.log(2/3)
        Pc_1 = numpy.log(1/3)
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
        Predicted_labels = numpy.argmax(SPost, axis=1) 

        # Accuracy and Error rate
        accuracy_array = []
        for i in range(0, K_validation_label_set.size):
            if Predicted_labels[i] == K_validation_label_set[i]:
                accuracy_array.append(1)
            else:
                accuracy_array.append(0)
        accuracy = numpy.array(accuracy_array).sum(0)/len(accuracy_array)
        error_rate = 1 - accuracy
        accuracy_final_list.append(accuracy)
        error_final_list.append(error_rate)

    accuracy_mean = (sum(accuracy_final_list)/K)*100
    error_mean = (sum(error_final_list)/K)*100
    print(accuracy_mean)
    print(error_mean)

MVG()
