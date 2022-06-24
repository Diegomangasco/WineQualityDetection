import numpy
import matplotlib.pyplot as plt
from load_data import *

def plot_hist(D, L):
    
    D0= D[:, L==0]
    D1= D[:, L==1]
     
    map_Di = {
        0: 'Fixed acidity',
        1: 'Volatile acidity',
        2: 'Citric acid',
        3: 'Residual sugar',
        4: 'Chlorides',
        5: 'Free sulfur dioxide',
        6: 'Total sulfur dioxide',
        7: 'Density',
        8: 'pH',
        9: 'Sulphates',
        10: 'Alcohol'        
    }
    
    for i in range(D.shape[0]):
        plt.figure()
        plt.xlabel(map_Di[i])
        plt.hist(D0[i, :], bins = 10, density = True, alpha = 0.4, label = 'Bad quality wine')
        plt.hist(D1[i, :], bins = 10, density = True, alpha = 0.4, label = 'Good quality wine')
        plt.legend()
        # plt.savefig('hist_%d.pdf' % i) 
       
    plt.show()
    
    
def plot_scatter(D, L):
    
    D0= D[:, L==0]
    D1= D[:, L==1]
     
    map_Di = {
        0: 'Fixed acidity',
        1: 'Volatile acidity',
        2: 'Citric acid',
        3: 'Residual sugar',
        4: 'Chlorides',
        5: 'Free sulfur dioxide',
        6: 'Total sulfur dioxide',
        7: 'Density',
        8: 'pH',
        9: 'Sulphates',
        10: 'Alcohol'        
    }
   
   
    for i in range(D.shape[0]):
        for j in range(D.shape[0]):
            if(i!=j):
                plt.figure()
                plt.xlabel(map_Di[i])
                plt.ylabel(map_Di[j])
                plt.plot(D0[i, :], D0[j, :],  linestyle='', marker='.', markersize=5, label = 'Bad quality wine')
                plt.plot(D1[i, :], D1[j, :],  linestyle='', marker='.', markersize=5, label = 'Good quality wine')
              
                plt.legend()

    plt.show()
    

if __name__=='__main__':
    TD, TL, _, _ = load_data()
    plot_hist(TD, TL)
    plot_scatter(TD, TL)
    
    
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    