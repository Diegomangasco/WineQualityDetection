from load_data import * 
from dimensionality_reduction import *
import numpy
import scipy.optimize

def mrow(array):
    return numpy.reshape(array, (1, array.size))

def mcol(array):
    return numpy.reshape(array, (array.size, 1))

def SVM_linear(DTR, LTR):

    D_hat = numpy.vstack([DTR, numpy.ones((1, DTR.shape[1]))])
    Z = numpy.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1
    C = 1

    H = numpy.dot(D_hat.T, D_hat)
    H = mcol(Z) * mrow(Z) * H

    def JDual(alpha):
        Ha = numpy.dot(H, mcol(alpha))
        aHa = numpy.dot(mrow(alpha), Ha)
        a1 = alpha.sum()
        return -0.5 * aHa.ravel() + a1, -Ha.ravel() + numpy.ones(alpha.size)

    def LDual(alpha):
        loss, grad = JDual(alpha)
        return -loss, -grad
    
    # def JPrimal(w):
    #     S = numpy.dot(mrow(w), D_hat)
    #     loss = numpy.maximum(numpy.zeros(S.shape), 1-Z*S).sum()
    #     return 0.5 * numpy.linalg.norm(w)**2 + C * loss

    alphaStar, _x, _y = scipy.optimize.fmin_l_bfgs_b(
        LDual,
        numpy.zeros(DTR.shape[1]),
        bounds = [(0,C)]*DTR.shape[1],
        factr=1.0,
        maxiter=100000,
        maxfun=100000,
    )

    wStar = numpy.dot(D_hat, mcol(alphaStar)*mcol(Z))

    return wStar

if __name__ == '__main__':
    prior= 4/9
    data = load_data()
    training_data = data[0]
    training_labels = data[1]
    
    training_data = computePCA(training_data, 9)

    K = 5
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

        wStar = SVM_linear(K_training_set, K_training_labels_set)
        K_validation_set_hat = numpy.vstack([K_validation_set, numpy.ones((1, K_validation_set.shape[1]))])
        S = numpy.dot(wStar.T, K_validation_set_hat)
        scores_list = numpy.concatenate((scores_list, S), axis=1)
        real_labels = numpy.concatenate((real_labels, K_validation_label_set), axis=0)

    scores_list = scores_list[:, 1:]
    ZTE = 2.0*real_labels - 1.0
    predicted = scores_list.ravel()
    acc = predicted[predicted*ZTE>0].shape[0]/predicted.shape[0]
    print(acc)
    


    
