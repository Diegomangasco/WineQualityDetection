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

def mean(class_identifier, training_data, training_labels):
    Dc = training_data[:, training_labels == class_identifier]
    return mcol(Dc.mean(1))

def covariance(class_identifier, training_data, training_labels):
    m = mean(class_identifier, training_data, training_labels)
    centered_matrix = training_data[:, training_labels == class_identifier] - m
    N = centered_matrix.shape[1]
    return numpy.dot(centered_matrix, centered_matrix.T)/N

def logpdf_GAU_ND(X, mean, covariance_matrix):
    M = X.shape[0];
    P = numpy.linalg.inv(covariance_matrix)
    const =  -0.5 * M * numpy.log(2*numpy.pi)
    const += -0.5 * numpy.linalg.slogdet(covariance_matrix)[1]
    
    l_x=[];
    for i in range(X.shape[1]):
       x = X[:, i:i+1]
       res= const - 0.5 * numpy.dot((x-mean).T, numpy.dot(P, (x-mean)))
       l_x.append(res)
     
    return mrow(numpy.array(l_x))

# def GMM_ll_perSample(X, gmm):
#     G = len(gmm)
#     N = X.shape[1]
#     S = numpy.zeros((G, N))
#     for g in range(G):
#         S[g, :] = logpdf_GAU_ND(X, gmm[g][1], gmm[g][2]) + numpy.log(gmm[g][0])
#     return scipy.special.logsumexp(S, axis=0)

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
        Sigma_tied= numpy.zeros((X.shape[0], X.shape[0]))
        Sigma_array= []
        Z_array=[]
        for g in range(G):
            gamma = P[g, :]
            Z = gamma.sum()
            F = (mrow(gamma)*X).sum(1)
            S = numpy.dot(X, (mrow(gamma)*X).T)
            mu = mcol(F/Z)
            Sigma = S/Z - numpy.dot(mu, mu.T)
            Z_array.append(Z)
            Sigma_array.append(Sigma)
        
        for g in range(G):
            Sigma_tied= Sigma_tied + Z_array[g]*Sigma_array[g]
        
        Sigma_tied= Sigma_tied/N
        
        U, s, _ = numpy.linalg.svd(Sigma_tied)
        psi=0.01
        s[s<psi] = psi
        Sigma_tied = numpy.dot(U, mcol(s)*U.T)
        
        for g in range(G):
            gamma = P[g, :]
            Z = gamma.sum()
            F = (mrow(gamma)*X).sum(1)
            S = numpy.dot(X, (mrow(gamma)*X).T)
            w = Z/N
            mu = mcol(F/Z)
            gmmNew.append((w, mu, Sigma_tied))
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

def weighted_logS(D, gmm_):
    log_S= numpy.zeros([1, D.shape[1]])
    for i in range(len(gmm_)):
        log_S= log_S+ gmm_[i][0]*logpdf_GAU_ND(D, gmm_[i][1], gmm_[i][2])
    return log_S

if __name__=='__main__':
    prior_array= [4/9, 1/5, 4/5]
    prior = prior_array[0]
    data = load_data()
    training_data = data[0]
    training_labels = data[1]
    test_data = data[2]
    test_labels = data[3]
    
    #P, training_data = computePCA(training_data, 9)
    #test_data = numpy.dot(P.T, test_data)

    components= 2
    iterations=0
        
    mean_0 = mean(0, training_data, training_labels)
    mean_1 = mean(1, training_data, training_labels)
    covariance_matrix_0 = covariance(0, training_data, training_labels)
    covariance_matrix_1 = covariance(1, training_data, training_labels)
    
    training_set_0=  training_data[:, training_labels==0]
    training_set_1=  training_data[:, training_labels==1]
    # compute the tied covariance
    N_0 = training_set_0.shape[1]
    N_1 = training_set_1.shape[1]
    tied_covariance = (1/training_data.shape[1])*(covariance_matrix_0*N_0 + covariance_matrix_1*N_1)
    
    gmm_array0= []
    gmm_array1= [] 
    gmm0=[]
    gmm1=[]
    
    # We need to create the triplets of (weight, mean, covariance) that will be given to the EM algorithm
    # The number of triplets depends on the number of component selected
    while iterations < numpy.log2(components):
        iterations= iterations + 1
        weight= 0
        comp= 2**iterations
        if iterations== 1:
            weight = 1.0/comp
            
            mean_vec0= numpy.zeros((mean_0.shape[0], comp))
            
            U0, s0, _ = numpy.linalg.svd(covariance_matrix_0)
            alpha0= 1
            d0 = U0[:, 0:1] * s0[0]**0.5 * alpha0
            
            U1, s1, _ = numpy.linalg.svd(covariance_matrix_1)
            alpha1= 1
            d1 = U1[:, 0:1] * s1[0]**0.5 * alpha1
            
            mean_vec0[:, 0]= (mcol(mean_0[:, 0]) + d0).ravel()
            mean_vec0[:, 1]= (mcol(mean_0[:, 0]) - d0).ravel()
            mean_vec1= numpy.zeros((mean_1.shape[0], comp))
            mean_vec1[:, 0]= (mcol(mean_1[:, 0]) + d1).ravel()
            mean_vec1[:, 1]= (mcol(mean_1[:, 0]) - d1).ravel()
            
            
            tied_cov_new= tied_covariance
            U, s, _ = numpy.linalg.svd(tied_cov_new)
            psi=0.01
            # Use a value psi that replaces the values toward zero in s
            s[s<psi] = psi
            tied_cov_new = numpy.dot(U, mcol(s)*U.T)
            
            for c in range(comp):
                gmm_array0.append((weight, mcol(mean_vec0[:, c]),  tied_cov_new))
            
            for c in range(comp):
                gmm_array1.append((weight, mcol(mean_vec1[:, c]),  tied_cov_new))
            
            gmm0= GMM_EM(training_set_0, gmm_array0)
            gmm1= GMM_EM(training_set_1, gmm_array1)
        else:
            gmm_array0= []
            gmm_array1= []
            weight0_array= numpy.zeros((comp))
            weight1_array= numpy.zeros((comp))
            d0_array= numpy.zeros((mean_0.shape[0], (int(comp/2))))
            d1_array= numpy.zeros((mean_1.shape[0], (int(comp/2))))
            cov_array0=[]
            cov_array1=[]
            
            for c in range(int(comp/2)):
                U0, s0, _ = numpy.linalg.svd(gmm0[c][2])
                cov_array0.append(gmm0[c][2])
                cov_array0.append(gmm0[c][2])
                alpha0= 1
                d0_array[:, c] = (U0[:, 0:1] * s0[0]**0.5 * alpha0).ravel()
            
            for c in range(int(comp/2)):
                U1, s1, _ = numpy.linalg.svd(gmm1[c][2])
                cov_array1.append(gmm1[c][2])
                cov_array1.append(gmm1[c][2])
                alpha1= 1
                d1_array[:, c] = (U1[:, 0:1] * s1[0]**0.5 * alpha1).ravel()
                
            for c in range(int(comp/2)):
                weight0_array[2*c]= gmm0[c][0]/2
                weight0_array[(2*c)+1]= gmm0[c][0]/2
                
            for c in range(int(comp/2)):
                weight1_array[2*c]= gmm1[c][0]/2
                weight1_array[(2*c)+1]= gmm1[c][0]/2
            
            mean_vec0= numpy.zeros((mean_0.shape[0], comp))
            
            for c in range(int(comp/2)):
                mean_vec0[:, 2*c]= (mcol(gmm0[c][1]) + mcol(d0_array[:, c])).ravel()
                mean_vec0[:, ((2*c)+1)]= (mcol(gmm0[c][1]) -  mcol(d0_array[:, c])).ravel()
            
            mean_vec1= numpy.zeros((mean_1.shape[0], comp))
            
            for c in range(int(comp/2)):
                mean_vec1[:, 2*c]= (mcol(gmm1[c][1]) +  mcol(d1_array[:, c])).ravel()
                mean_vec1[:, ((2*c)+1)]= (mcol(gmm1[c][1]) -  mcol(d1_array[:, c])).ravel()
                
            for c in range(comp):
                gmm_array0.append((weight0_array[c], mcol(mean_vec0[:, c]), cov_array0[c]))
            for c in range(comp):
                gmm_array1.append((weight1_array[c], mcol(mean_vec1[:, c]), cov_array1[c]))
            
            gmm0= GMM_EM(training_set_0, gmm_array0)
            gmm1= GMM_EM(training_set_1, gmm_array1)
    
    weighted_logS0= weighted_logS(test_data, gmm0)
    weighted_logS1= weighted_logS(test_data, gmm1)
    logS= numpy.concatenate((weighted_logS0, weighted_logS1), axis=0)
    logS= logS.T

    Pc_0 = numpy.log(1-prior)
    Pc_1 = numpy.log(prior)
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
    
   
    # Compute the confusion matrix, accurancy and error rates
    confusion = numpy.zeros([2,2], dtype=int)
    for j in range(2):
        for i in range(2):
            confusion[j,i] = ((Predicted_labels==j) * (test_labels==i)).sum()
    
 
    FNR_ = confusion[0,1]/(confusion[0,1]+ confusion[1,1])
    FPR_ = confusion[1,0]/(confusion[1,0]+ confusion[0,0])
    err = (confusion[0,1]+confusion[1,0])/(confusion[0,1]+confusion[1,1]+confusion[1,0]+confusion[0,0])
    acc = 1-err
    print("Model accuracy:", round(acc*100, 3), "%")
    print("Model error:", round(err*100, 3), "%")
    print("Confusion matrix: ") 
    print(confusion)

    # Compute the log-lokelihood ratio llr by using the score matrix
    S = numpy.exp(logS)
    llr = numpy.zeros([S.shape[0]])
    for i in range(logS.shape[0]):    
        llr[i] = numpy.log(S[i,1]/S[i,0])

    # to create a file with the scores
    llr_tied_gmm_for_file= numpy.asarray(llr)
    numpy.save('llr_tied_gmm.npy', llr_tied_gmm_for_file)
        
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
    confusion__matrix = conf_matrix(llr, test_labels, prior, Cfn, Cfp)
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
                Conf[j, i] = ((Pred==j) * (test_labels==i)).sum()
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
        c_m = conf_matrix(llr, test_labels, eff_prior, Cfn, Cfp)
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
                    Conf[j, i]= ((Pred==j) * (test_labels==i)).sum()
            FNR_minDCF = Conf[0,1] / (Conf[0,1] + Conf[1,1])
            FPR_minDCF = Conf[1,0] / (Conf[1,0] + Conf[0,0])
            Bemp_minDCF[idx]= eff_prior*Cfn*FNR_minDCF + (1-eff_prior)*Cfp*FPR_minDCF
             
        Bemp_min= Bemp_minDCF.min()   
        
        minDCF[index]= Bemp_min/Bdummy
        
        
    matplotlib.pyplot.xlabel('log(π/(1-π))')
    matplotlib.pyplot.ylabel('DCF')
    matplotlib.pyplot.title('Bayes error plot')
    matplotlib.pyplot.plot(effPriorLogOdds, normalizedDCF, color='r',  label= 'GMM, 2 components, with PCA(m=9) - act DCF')
    matplotlib.pyplot.plot(effPriorLogOdds, minDCF, color='b', label= 'GMM, 2 components, no PCA(m=9) - min DCF')
    matplotlib.pyplot.ylim([0, 1.1])
    matplotlib.pyplot.xlim([-3, 3])
    matplotlib.pyplot.legend()
    matplotlib.pyplot.show()

       
       
        

