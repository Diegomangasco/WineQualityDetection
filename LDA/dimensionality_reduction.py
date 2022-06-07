import numpy 
import scipy.linalg

def mcol(v):
    return v.reshape((v.size, 1))

def computeCovarianceForEachClass(matrix, categories):
    data = numpy.array(matrix)
    D0 = data[:, categories==0]
    D1 = data[:, categories==1]
    mu0 = D0.mean(1)
    mu1 = D1.mean(1)
    N = data.shape[1]
    N0 = D0.shape[1]
    N1 = D1.shape[1]
    CD0 = D0 - mcol(mu0)
    CD1 = D1 - mcol(mu1)
    CM0 = numpy.dot(CD0, CD0.T)/N0
    CM1 = numpy.dot(CD1, CD1.T)/N1
    result = (N0*CM0+N1*CM1)/N
    return result

def computeCovarianceBetweenClasses(matrix, categories):
    data = numpy.array(matrix)
    D0 = data[:, categories==0]
    D1 = data[:, categories==1]
    mu = data.mean(1)
    mu0 = D0.mean(1)
    mu1 = D1.mean(1)
    N = data.shape[1]
    N0 = D0.shape[1]
    N1 = D1.shape[1]
    CD = data - mcol(mu)
    CD0 = D0 - mcol(mu0)
    CD1 = D1 - mcol(mu1)
    CM = numpy.dot(CD, CD.T)/N
    CM0 = numpy.dot(CD0, CD0.T)/N0
    CM1 = numpy.dot(CD1, CD1.T)/N1
    result = (N0*(CM-CM0)+N1*(CM-CM1))/N
    return result 

def computePCA(data, dimensions):
    data = numpy.array(data)
    mu = data.mean(1)
    centeredData = data - numpy.reshape(mu, (mu.size, 1))
    covarianceMatrix = numpy.dot(centeredData, centeredData.T)/data.shape[1]
    U, _, _ = numpy.linalg.svd(covarianceMatrix)
    # Take all the columns in the reverse order (-1), and then takes only the first columns
    P= U[:, 0:dimensions]
    DP = numpy.dot(P.T, data)
    return DP

def computeLDA(data, labels, dimensions):
    covarianceWithinClasses = computeCovarianceForEachClass(data, labels)
    covarianceBetweenClasses = computeCovarianceBetweenClasses(data, labels)
    s, U = scipy.linalg.eigh(covarianceBetweenClasses, covarianceWithinClasses)
    W = U[:, ::-1][:, 0:dimensions]
    UW, _, _ = numpy.linalg.svd(W)
    U = UW[:, 0:dimensions]
    DP = numpy.dot(U.T, data)
    return DP, U
