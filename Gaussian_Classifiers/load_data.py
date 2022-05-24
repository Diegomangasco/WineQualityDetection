import numpy
import numpy
import matplotlib
import matplotlib.pyplot as plt

def mcol(array):
    return numpy.reshape(array, (array.shape[0], 1))    # Reshape as column array

def mrow(array):
    return numpy.reshape(array, (1, array.shape[0]))    # Reshape as row array

def load_file(input_file):
    file = open(input_file, 'r')
    categories = []
    data = []
    # Read the file
    for line in file.readlines():
        try:
            # Take the first eleven columns that represent the wine features
            features = mcol(numpy.array(line.split(',')[0:11]))
            # Take the last column that is the wine quality
            category = line.split(',')[-1].strip()
            data.append(features)
            categories.append(category)
        except:
            pass   
    file.close()
    return numpy.hstack(data).astype(numpy.float32), numpy.array(categories, dtype=numpy.int32)

def load_data():
    # Take the training and the test datasets files, passed as arguments to the script
    training_data = './Train.txt'
    test_data = './Test.txt'
    training_set = load_file(training_data)
    test_set = load_file(test_data)
    # Return the values of wine features and categories for both training and test set to the caller
    training_data = training_set[0]
    training_labels = training_set[1]
    test_data = test_set[0]
    test_labels = test_set[1]
    return training_data, training_labels, test_data, test_labels 


def plot_hist(D, L):
    
    D0= D[:, L==0]
    D1= D[:, L==1]
     
    map_Di = {
        0: 'First feature',
        1: 'Second feature',
        2: 'Third feature',
        3: 'Fourth feature',
        4: 'Fifth feature',
        5: 'Sixth feature',
        6: 'Seventh feature',
        7: 'Eighth feature',
        8: 'Nineth feature',
        9: 'Tenth feature',
        10: 'Eleventh feature'      
    }
    
    for i in range(D.shape[0]):
        plt.figure()
        plt.xlabel(map_Di[i])
        plt.hist(D0[i, :], bins = 10, density = True, alpha = 0.4, label = 'Bad wine')
        plt.hist(D1[i, :], bins = 10, density = True, alpha = 0.4, label = 'Good wine')
        plt.legend()
        # plt.savefig('hist_%d.pdf' % i) 
       
    plt.show()
    
    
def plot_scatter(D, L):
    
    D0= D[:, L==0]
    D1= D[:, L==1]
     
    map_Di = {
        0: 'First feature',
        1: 'Second feature',
        2: 'Third feature',
        3: 'Fourth feature',
        4: 'Fifth feature',
        5: 'Sixth feature',
        6: 'Seventh feature',
        7: 'Eighth feature',
        8: 'Nineth feature',
        9: 'Tenth feature',
        10: 'Eleventh feature'      
    }
   
    
    for i in range(D.shape[0]):
        for j in range(D.shape[0]):
            if(i!=j):
                plt.figure()
                plt.xlabel(map_Di[i])
                plt.ylabel(map_Di[j])
                plt.plot(D0[i, :], D0[j, :],  linestyle='', marker='.', markersize=5, label = 'Bad wine')
                plt.plot(D1[i, :], D1[j, :],  linestyle='', marker='.', markersize=5, label = 'Good wine')
              
                plt.legend()

    plt.show()
    

def dataset_mean(D):
    data_mean = D.mean(1)
    return mcol(data_mean)


def centered_dataset(D):
    centered_D= D- dataset_mean(D)   
    return centered_D 

def covariance_matrix(Dc):
    return (1/Dc.shape[1])*numpy.dot(Dc, Dc.T)

def singular_value_decomposition(covariance_matrix):
    return numpy.linalg.svd(covariance_matrix) #In this case, the singular values (which are equal to the eigenvalues) 
    #are sorted in descending order, and the columns of U are the corresponding eigenvectors

# def p_from_svd(covariance_matrix):
#     U, _, _= singular_value_decomposition(covariance_matrix)
#     return  U[:, 0:2] # in this case m=2
    
if __name__=='__main__':
    TD, TL, _, _ = load_data()
    # plot_hist(TD, TL)
    # plot_scatter(TD, TL)
    mu= dataset_mean(TD)
    TDc= centered_dataset(TD)
    C= covariance_matrix(TDc)
    U, Epsilon, _= singular_value_decomposition(C)
    # Epsilon contains the eigenvalues in descending order
    # print(U)
    sum_of_all_eigenvalues= 0
   
    # t= 0.95
    
    for i in range(Epsilon.shape[0]):
        sum_of_all_eigenvalues+= Epsilon[i]
    
    sum_of_m_eigenvalues=[]
    
    for i in range(Epsilon.shape[0]):
        temp=0
        for j in range(i):
            temp+= Epsilon[j]
        sum_of_m_eigenvalues.append(temp) 
            
    sum_of_m_eigenvalues.append(sum(Epsilon))
    
    sum_of_m_eigenvalues.pop(0)
    
    v_sum_of_m_eigenvalues= numpy.array(sum_of_m_eigenvalues)
    
    quotient=[]
    for i in range(Epsilon.shape[0]):
        quotient.append(v_sum_of_m_eigenvalues[i]/sum_of_all_eigenvalues)
     
    
    print(Epsilon)
    print(quotient)
    # --------------
    P= U[:, 0:2] # m=2
    y= numpy.dot(P.T, TD)
    y0= y[:, TL==0]
    y1= y[:, TL==1]
    
    plt.figure()
    plt.plot(y0[0], y0[1], linestyle='', marker='.', markersize=3, label = 'Bad wine')
    plt.plot(y1[0], y1[1], linestyle='', marker='.', markersize=3, label = 'Good wine')
    plt.legend()

    plt.show()
    
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    