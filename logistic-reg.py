import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

def sigmoid(x):
    return 1/(1 + np.exp(-x))
def costFunc(theta,x,y):
    m = x.shape[0]
    J = (-1/m) * np.sum(np.multiply(y, np.log(sigmoid(x @ theta))) 
        + np.multiply((1-y), np.log(1 - sigmoid(x @ theta))))
    return J
def gradient(theta, x, y):
    m = x.shape[0]
    return ((1/m) * x.T @ (sigmoid(x @ theta) - y))

data = pd.read_csv('ex2data1.txt', header = None)
x = data.iloc[:,0:2]
y = data.iloc[:,2]

#mask = y == 1
#adm = plt.scatter(x[mask][0].values, x[mask][1].values)
#not_adm = plt.scatter(x[~mask][0].values, x[~mask][1].values)

m,n = x.shape
x= np.hstack((np.ones((m,1)),x))
y = y[:, np.newaxis]
theta = np.zeros((n+1,1))

j = costFunc(theta,x,y)
print(j)

temp = opt.fmin_tnc(func = costFunc, 
                    x0 = theta.flatten(),fprime = gradient, 
                    args = (x, y.flatten()))
theta_optimized = temp[0]
print(theta_optimized)

j = costFunc(theta_optimized[:, np.newaxis],x,y)
print(j)