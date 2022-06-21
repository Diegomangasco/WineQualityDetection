import numpy
import scipy
import sklearn
import numpy.linalg
import scipy.optimize
import sklearn.datasets
import pylab
import matplotlib
from load_data import *
from dimensionality_reduction import *

def vcol(v):
    return v.reshape((v.size, 1))

def vrow(v):
    return v.reshape((1, v.size))

def logreg_obj_wrap(DTR, LTR, l, prior_tr):
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
        return 0.5*l*numpy.linalg.norm(w)**2 + prior_tr*cxe_one.mean() + (1-prior_tr)*cxe_minus_one.mean()
    return logreg_obj
    
def conf_matrix(scores, labs, pr, C_fn, C_fp):
    
    # Computing c* comparing the llr with a threshold t
    t = - numpy.log((pr*C_fn)/((1-pr)*C_fp))
    
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

if __name__=='__main__':
    prior_array = [4/9, 1/5, 4/5]
    prior_array[0]   # prior for training the model
    prior_array[0]   # prior for evaluate the model
    data = load_data()
    training_data = data[0]
    training_labels = data[1]
    test_data = data[2]
    test_labels = data[3]

    P, training_data = computePCA(training_data, 9)
    test_data = numpy.dot(P.T, test_data)
    
    lamb = 1e-4;

    # Use the scipy function to compute LBFGS method for minimization
    x0 = numpy.zeros(training_data.shape[0] + 1)
    logreg_obj = logreg_obj_wrap(training_data, training_labels, lamb, prior_t)
    minimum_position, function_value, dictionary = scipy.optimize.fmin_l_bfgs_b(logreg_obj, x0, approx_grad=True)
    w_for_minimum = minimum_position[0:-1]
    b_for_minimum = minimum_position[-1]
    # Compute S 
    w_for_minimum = numpy.reshape(w_for_minimum, (1, w_for_minimum.shape[0]))
    # Compute the scores
    S = numpy.dot(w_for_minimum, test_data) + b_for_minimum
    
    
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
    
    
    #This scores are NOT the scores of the LR, but they are minus the log-odds to calaculate DCF
    Scores = numpy.array(S[0, :]) - numpy.log( prior_t/ (1-(prior_t)) )
    
        
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