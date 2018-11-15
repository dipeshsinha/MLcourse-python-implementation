import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

def sigmoid(x):
    return 1/(1+np.exp(-x))

def costFunc(nn_params, input_layer_size, hidden_layer_size, num_labels, x, y, lmbda):
    theta1 = np.reshape(nn_params[:hidden_layer_size*(input_layer_size+1)], (hidden_layer_size, input_layer_size+1), 'F')
    theta2 = np.reshape(nn_params[hidden_layer_size*(input_layer_size+1):], (num_labels, hidden_layer_size+1), 'F')
    
    a1 = np.hstack((np.ones((len(y), 1)), x))
    a2 = sigmoid(a1 @ theta1.T)
    a2 = np.hstack((np.ones((len(y), 1)), a2))
    h = sigmoid(a2 @ theta2.T)
    y_d = pd.get_dummies(y.flatten())
    

data = loadmat('ex4data1.mat')
x = data['X']
y = data['y']

# visualizing the data
#_, axarr = plt.subplots(10,10,figsize=(10,10))
#for i in range(10):
#    for j in range(10):
#       axarr[i,j].imshow(x[np.random.randint(x.shape[0])].\
#reshape((20,20), order = 'F'))          
#       axarr[i,j].axis('off')
       
       
weights = loadmat('ex4weights.mat')
theta1 = weights['Theta1']
theta2 = weights['Theta2']

nn_params = np.hstack((theta1.ravel(order='F'), theta2.ravel(order='F')))

input_layer_size = 400
hidden_layer_size = 25
num_labels = 10
lmbda = 1

costFunc(nn_params, input_layer_size, hidden_layer_size, num_labels, x, y, lmbda)