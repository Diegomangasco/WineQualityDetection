import numpy 
import scipy

def mcol(v):
    return v.reshape((v.size, 1))

def computeCovarianceForEachClass(matrix, categories):
    data = numpy.array(matrix)
    D1 = data[:, categories==0]
    D2 = data[:, categories==1]
    mu1 = D1.mean(1)
    mu2 = D2.mean(1)
    CD1 = D1 - mcol(mu1)
    CD2 = D2 - mcol(mu2)
    CM1 = numpy.dot(CD1, CD1.T)
    CM2 = numpy.dot(CD2, CD2.T)
    result = (CM1 + CM2)/data.shape[1]
    return result

def computeCovarianceBetweenClasses(matrix, categories):
    data = numpy.array(matrix)
    D1 = data[:, categories==0]
    D2 = data[:, categories==1]
    mu1 = D1.mean(1)
    mu2 = D2.mean(1)
    mu = data.mean(1)
    dataForEachClass = data.shape[1]/2
    mean1 = dataForEachClass * numpy.dot((mcol(mu1)-mcol(mu)), (mcol(mu1)-mcol(mu)).T)
    mean2 = dataForEachClass * numpy.dot((mcol(mu2)-mcol(mu)), (mcol(mu2)-mcol(mu)).T)
    result = (mean1 + mean2)/data.shape[1]
    return result 

def computePCA(data, dimensions):
    data = numpy.array(data)
    mu = data.mean(1)
    centeredData = data - numpy.reshape(mu, (mu.size, 1))
    covarianceMatrix = numpy.dot(centeredData, centeredData.T)/data.shape[1]
    s, U = numpy.linalg.eigh(covarianceMatrix)
    # Take all the columns in the reverse order (-1), and then takes only the first columns
    P = U[:, ::-1][:, 0:dimensions]  
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
    return DP