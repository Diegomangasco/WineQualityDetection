import numpy
import scipy
import sklearn
import numpy.linalg
import scipy.optimize
import sklearn.datasets
from load_data import *
from dimensionality_reduction import *

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
        # cxe= numpy.logaddexp(0, -S*Z) # cross-entropy
        
        # For the prior weighted version of the model
        cxe_one = numpy.logaddexp(0, -S[:, Z>0]*Z[Z>0])
        cxe_minus_one = numpy.logaddexp(0, -S[:, Z<0]*Z[Z<0])
        return 0.5*l*numpy.linalg.norm(w)**2 + (4/9)*cxe_one.mean() + (5/9)*cxe_minus_one.mean()
    return logreg_obj
    

if __name__=='__main__':
    data = load_data()
    training_data = data[0]
    training_labels = data[1]
    training_data = computePCA(training_data, 8)
    
    lamb = 0.0001;
    K = 10

    accuracy_final_list = []
    error_final_list = []
    
    length_of_interval = int(training_data.shape[1]/K)
    index = 0
    # For cycle for doing K fold cross validation 
    while index <= (training_data.shape[1] - length_of_interval):
        # Take one section as validation set
        start = index
        end = index + length_of_interval
        K_validation_set = training_data[:, start:end]
        K_validation_label_set = training_labels[start:end]
        K_training_set_part1 = training_data[:, 0:start]
        K_training_set_part2 = training_data[:, end:]
        K_training_set = numpy.concatenate((K_training_set_part1, K_training_set_part2), axis=1)
        K_training_labels_set_part1 = training_labels[0:start]
        K_training_labels_set_part2 = training_labels[end:]
        K_training_labels_set = numpy.concatenate((K_training_labels_set_part1, K_training_labels_set_part2), axis=None)

        index = index + length_of_interval

        x0 = numpy.zeros(K_training_set.shape[0] + 1)
        logreg_obj = logreg_obj_wrap(K_training_set, K_training_labels_set, lamb)
        minimum_position, function_value, dictionary = scipy.optimize.fmin_l_bfgs_b(logreg_obj, x0, approx_grad=True)
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
        accuracy_final_list.append(accuracy)
        print("Finished a fold")

    accuracy_mean = (sum(accuracy_final_list)/K)*100
    error_mean = 100-accuracy_mean
    print("Model accuracy:", accuracy_mean)
    print("Model error:", error_mean)
