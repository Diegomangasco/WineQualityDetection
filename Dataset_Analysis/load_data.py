import numpy
import numpy

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
  
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
