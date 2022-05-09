import numpy
import scipy
import sklearn
import numpy.linalg
import scipy.optimize
import sklearn.datasets
from load_data import *
from dimensionality_reduction import *

class logRegClass:
    def __init__(self, DTR, LTR, l):
        self.DTR = DTR
        self.LTR = LTR
        self.l = l
    def logreg_obj(self, v):
        w, b = v[0:-1], v[-1]
        w = numpy.reshape(w, (1, w.size))
        constant = (self.l/2)*(numpy.linalg.norm(w)**2)
        counter = 0
        log_vector = []
        for i in range(0, len(self.LTR)):
            sample = numpy.reshape(self.DTR[:, i], (self.DTR.shape[0], 1))
            S = numpy.dot(w, sample) + b
            if self.LTR[i] == 0:
                zi = -1
            else:
                zi = 1
            log_vector.append(numpy.logaddexp(0, -zi*S))
            counter += 1
        return constant + (1/counter)*numpy.sum(log_vector)

def logistic_regression():
    data = load_data()
    training_data = data[0]
    training_labels = data[1]
    #training_data = computePCA(training_data, 8)
    #training_data = computeLDA(training_data, training_labels, 5)
    l = 0.01;
    K = 3
    K_fold_set = numpy.hsplit(training_data, K)
    K_fold_set = numpy.array(K_fold_set)
    K_fold_labels = numpy.split(training_labels, K)

    accuracy_final_list = []
    error_final_list = []
    
    for i in range(0, K):
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
        
        logRegObj = logRegClass(K_training_set, K_training_labels_set, l)
        x0 = numpy.zeros(K_training_set.shape[0] + 1)
        # You can now use logRegObj.logreg_obj as objective function:
        minimum_position, function_value, dictionary = scipy.optimize.fmin_l_bfgs_b(logRegObj.logreg_obj, x0, maxfun=20000, maxiter=20000, approx_grad=True)
        w_for_minimum = minimum_position[0:-1]
        b_for_minimum = minimum_position[-1]
        # Compute S 
        w_for_minimum = numpy.reshape(w_for_minimum, (1, w_for_minimum.shape[0]))
        S = numpy.dot(w_for_minimum, K_validation_set) + b_for_minimum
        # Predicted labels

        Predicted_labels = []
        for i in range(0, S.shape[1]):
            if S[0][i] > 0:
                Predicted_labels.append(1)
            else:
                Predicted_labels.append(0)

        accuracy_array = []
        for j in range(0, K_validation_label_set.size):
            if Predicted_labels[j] == K_validation_label_set[j]:
                accuracy_array.append(1)
            else:
                accuracy_array.append(0)

        accuracy = numpy.array(accuracy_array).sum(0)/len(accuracy_array)
        error_rate = 1 - accuracy
        accuracy_final_list.append(accuracy)
        error_final_list.append(error_rate)
        print("Finished a fold")

    accuracy_mean = (sum(accuracy_final_list)/K)*100
    error_mean = (sum(error_final_list)/K)*100
    print(accuracy_mean)
    print(error_mean)

logistic_regression()