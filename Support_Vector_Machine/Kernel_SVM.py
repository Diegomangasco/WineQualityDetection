from load_data import * 
from dimensionality_reduction import *
import numpy
import scipy.optimize

def SVM_exp_kernel(DTR, LTR, gamma, C, k):
        
    Z = numpy.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1

    Dist = mcol((DTR**2).sum(0)) + mrow((DTR**2).sum(0)) - 2*numpy.dot(DTR.T, DTR)
    print(Dist.shape)
    H = numpy.exp(-gamma*Dist) + k
    H = mcol(Z)*mrow(Z)*H

    def JDual(alpha):   # alpha=values of Lagrange multipliers
        Ha = numpy.dot(H, mcol(alpha))
        aHa = numpy.dot(mrow(alpha), Ha)
        a1 = alpha.sum()
        return -0.5 * aHa.ravel() + a1, -Ha.ravel() + numpy.ones(alpha.size) # grad=-H*alpha+1

    def LDual(alpha):   # function that we actually minimize -> -loss and -grad
        loss, grad = JDual(alpha)
        return -loss, -grad

    alphaStar, _x, _y = scipy.optimize.fmin_l_bfgs_b(
        LDual,
        numpy.zeros(DTR.shape[1]), # initial value are zeros
        bounds = [(0,C)]*DTR.shape[1],  # values between which the alphas values must remain
        factr=1.0,
        maxiter=100000,
        maxfun=100000,
    )

    wStar = numpy.dot(Dist, mcol(alphaStar)*mcol(Z)) # dual solution w=sum(alpha_i*Z_i*X_i)

    return wStar

if __name__ == '__main__':
    prior= 4/9
    data = load_data()
    training_data = data[0]
    training_labels = data[1]
    
    training_data = computePCA(training_data, 9)

    K = 5
    C = 1
    k_SVM = 1
    gamma = 1
    real_labels = []
    scores_list = numpy.zeros([1, 1])
    
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

        wStar = SVM_exp_kernel(K_training_set, K_training_labels_set, gamma, C, k_SVM)
        K_validation_set_hat = numpy.vstack([K_validation_set, numpy.ones((1, K_validation_set.shape[1]))*k_SVM])
        S = numpy.dot(wStar.T, K_validation_set_hat)
        scores_list = numpy.concatenate((scores_list, S), axis=1)
        real_labels = numpy.concatenate((real_labels, K_validation_label_set), axis=0)

    scores_list = scores_list[:, 1:]
    ZTE = 2.0*real_labels - 1.0
    predicted = scores_list.ravel()
    acc = predicted[predicted*ZTE>0].shape[0]/predicted.shape[0]
    print(acc)
    