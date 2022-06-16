from load_data import * 
from dimensionality_reduction import *
import numpy
import scipy.optimize
import pylab
import matplotlib

def mrow(array):
    return numpy.reshape(array, (1, array.size))

def mcol(array):
    return numpy.reshape(array, (array.size, 1))

def conf_matrix(scores, labs, pr, C_fn, C_fp):
    
    # Computing c* comparing the llr with a threshold t
    t = - numpy.log(pr*C_fn)/((1-pr)*C_fp)
    
    C_star=numpy.zeros([scores.shape[0],], dtype= int)
    
    for i in range(scores.shape[0]):
        if scores[i]>t:
            C_star[i] = 1
        else:
            C_star[i] = 0
            
    # Computing the confusion matrix, comparing labels and c*    
    conf_matr= numpy.zeros([2,2], dtype= int)
    for j in range(2):
        for i in range(2):
            conf_matr[j,i]= ((C_star==j) * (labs==i)).sum()
    return conf_matr

def SVM_linear(DTR, LTR, C, k):

    D_hat = numpy.vstack([DTR, numpy.ones((1, DTR.shape[1]))*k])    # simulate the effect of some bias
    
    Z = numpy.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1

    H = numpy.dot(D_hat.T, D_hat) 
    H = mcol(Z) * mrow(Z) * H   # Hij = Zi Zj Xi.T Xj

    def JDual(alpha):   # alpha=values of Lagrange multipliers
        Ha = numpy.dot(H, mcol(alpha))
        aHa = numpy.dot(mrow(alpha), Ha)
        a1 = alpha.sum()
        return -0.5 * aHa.ravel() + a1, -Ha.ravel() + numpy.ones(alpha.size) # grad= - H alpha + 1

    def LDual(alpha):   # function that we actually minimize -> -loss and -grad
        loss, grad = JDual(alpha)
        return -loss, -grad
    
    # only to verify if the dual is computed well (computing the duality gap)
    # def JPrimal(w):
    #     S = numpy.dot(mrow(w), D_hat)
    #     loss = numpy.maximum(numpy.zeros(S.shape), 1-Z*S).sum()
    #     return 0.5 * numpy.linalg.norm(w)**2 + C * loss

    alphaStar, _x, _y = scipy.optimize.fmin_l_bfgs_b(
        LDual,
        numpy.zeros(DTR.shape[1]), # initial value are zeros
        bounds = [(0,C)]*DTR.shape[1],  # values between which the alphas values must remain
        factr=1.0,
        maxiter=100000,
        maxfun=100000,
    )

    wStar = numpy.dot(D_hat, mcol(alphaStar)*mcol(Z)) # primal solution: w=sum(alpha_i Z_i X_i)
    
    return wStar

if __name__ == '__main__':
    prior_array= [4/9, 1/5, 4/9] 
    # prior_t = prior_array[0]   # prior for training the model
    prior_tilde = prior_array[0]   # prior for evaluate the model
    data = load_data()
    training_data = data[0]
    training_labels = data[1]
    test_data = data[2]
    test_labels = data[3]
    
    # training_data = computePCA(training_data, 7)
    # test_data = computePCA(test_data, 7)

    # parameters for SVM
    C = 1
    k_SVM = 1
    
    w_star = SVM_linear(training_data, training_labels, C, k_SVM)
    K_validation_set_hat = numpy.vstack([test_data, numpy.ones((1, test_data.shape[1]))*k_SVM])
    S = numpy.dot(w_star.T, K_validation_set_hat)
    
    pred_labels = []
    for i in range(0, test_data.shape[1]):
        if S[0][i] > 0:
            pred_labels.append(1)
        else:
            pred_labels.append(0)
    pred_labels = numpy.array(pred_labels)
    # Compute the confusion matrix, accurancy and error rates
    confusion = numpy.zeros([2,2], dtype=int)
    for j in range(2):
        for i in range(2):
            confusion[j,i] = ((pred_labels==j) * (test_labels==i)).sum()
 
    FNR_ = confusion[0,1]/(confusion[0,1] + confusion[1,1])
    FPR_ = confusion[1,0]/(confusion[1,0] + confusion[0,0])
    err = (confusion[0,1]+confusion[1,0])/(confusion[0,1]+confusion[1,1]+confusion[1,0]+confusion[0,0])
    acc = 1-err
    print("Model accuracy:", round(acc*100, 3), "%")
    print("Model error:", round(err*100, 3), "%")
    print("Confusion matrix: ") 
    print(confusion)
    
    Scores = numpy.array(S[0, :]) 

    # Compute the calcusus for the ROC diagram
    thresholds_ROC = numpy.array(Scores)
    thresholds_ROC.sort()
    thresholds_ROC = numpy.concatenate([numpy.array([-numpy.inf]), thresholds_ROC, numpy.array([numpy.inf])])
    FPR = numpy.zeros(thresholds_ROC.size)
    TPR = numpy.zeros(thresholds_ROC.size)
    
    for idx, t in enumerate(thresholds_ROC):
        Pred = numpy.int32(Scores > t)
        Conf = numpy.zeros((2, 2))
        for j in range(2):
            for i in range(2):
                # Confusion matrix for each threshold
                Conf[j, i] = ((Pred==j) * (test_labels==i)).sum()
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
    confusion__matrix = conf_matrix(Scores, test_labels, prior_tilde, Cfn, Cfp)
    FNR_DCF = confusion__matrix[0,1] / ( confusion__matrix[0,1] + confusion__matrix[1,1])
    FPR_DCF = confusion__matrix[1,0] / ( confusion__matrix[1,0] + confusion__matrix[0,0])
    # Bayes empirical risk
    Bemp = prior_tilde*Cfn*FNR_DCF + (1-prior_tilde)*Cfp*FPR_DCF
    # Bayes empirical risk with a dummy strategy
    Bdummy = min(prior_tilde*Cfn, (1-prior_tilde)*Cfp) 
    # Normalized DCF
    normDCF = Bemp/Bdummy
    print("Actual normalized DCF", round(normDCF, 3))

     # Compute the minimum normalized DCF for our model
    thresholds = numpy.array(Scores)
    Bempirical = numpy.zeros(thresholds.size)
    for idx, t in enumerate(thresholds):
        Pred = numpy.int32(Scores > t)
        Conf = numpy.zeros((2, 2))
        for j in range(2):
            for i in range(2):
                Conf[j, i] = ((Pred==j) * (test_labels==i)).sum()
        FNR_minDCF = Conf[0,1] / (Conf[0,1] + Conf[1,1])
        FPR_minDCF = Conf[1,0] / (Conf[1,0] + Conf[0,0])
        Bempirical[idx]= prior_tilde*Cfn*FNR_minDCF + (1-prior_tilde)*Cfp*FPR_minDCF
   
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
        c_m = conf_matrix(Scores, test_labels, eff_prior, Cfn, Cfp)
        FNR_actualDCF =  c_m[0,1] / (c_m[0,1] + c_m[1,1])
        FPR_actualDCF =  c_m[1,0] / (c_m[1,0] + c_m[0,0])
        Bemp_actualDCF = eff_prior*Cfn*FNR_actualDCF + (1-eff_prior)*Cfp*FPR_actualDCF
        normBemp= Bemp_actualDCF/Bdummy
        normalizedDCF[index]= normBemp
        
        # Compute the min DCF
        Bemp_minDCF= numpy.zeros(thresholds.size)
        for idx, t in enumerate(thresholds):
            Pred = numpy.int32(thresholds > t)
            Conf = numpy.zeros((2, 2))
            for j in range(2):
                for i in range(2):
                    Conf[j, i]= ((Pred==j) * (test_labels==i)).sum()
            FNR_minDCF = Conf[0,1] / (Conf[0,1] + Conf[1,1])
            FPR_minDCF = Conf[1,0] / (Conf[1,0] + Conf[0,0])
            Bemp_minDCF[idx]= eff_prior*Cfn*FNR_minDCF + (1-eff_prior)*Cfp*FPR_minDCF
             
        Bemp_min= Bemp_minDCF.min()   
        
        minDCF[index]= Bemp_min/Bdummy
        
        
    matplotlib.pyplot.xlabel('log(π/(1-π))')
    matplotlib.pyplot.ylabel('DCF')
    matplotlib.pyplot.title('Bayes error plot')
    matplotlib.pyplot.plot(effPriorLogOdds, normalizedDCF, color='r',  label= 'actual DCF')
    matplotlib.pyplot.plot(effPriorLogOdds, minDCF, color='b', label= 'min DCF')
    matplotlib.pyplot.ylim([0, 1.1])
    matplotlib.pyplot.xlim([-3, 3])
    matplotlib.pyplot.legend()
    matplotlib.pyplot.show()

    
    



    
