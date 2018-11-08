import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

def sigmoid(x):
  return 1/(1+np.exp(-x))

def features(x1, x2):
    temp = np.ones(x1.shape[0])[:,np.newaxis]
    for i in range(1, 7):
        for j in range(1, i+1):
            temp = np.hstack((temp, np.multiply(np.power(x1, i-j), np.power(x2, j))[:,np.newaxis]))
    return temp

def costFunc(theta, x, y, lam):
    m = len(y)
    c1 = (-1/m)*(np.sum((y.T @ np.log(sigmoid(x @ theta)))+((1-y.T)@np.log(1-sigmoid(x @ theta)))))
    q = (lam/(2*m))*(theta[1:].T @ theta[1:])
    c = c1 + q
    return c

def gradient(theta,x,y,lmbda):
    m = len(y)
    c = np.zeros([m,1])
    c = (1/m)*(x.T @ (sigmoid(x@theta)-y))
    c[1:] = c[1:] + ((lmbda/m)*theta[1:])
    return c

data = pd.read_csv('ex2data2.txt', header = None)
x = data.iloc[:,:-1]
y = data.iloc[:,2]

#print(data.head())

# visualising data

#mask = y == 1
#pass1 = plt.scatter(x[mask][0], x[mask][1])
#fail1 = plt.scatter(x[~mask][0], x[~mask][1])
#plt.show()

#increase number of features to make non linear decision boundary

x = features(x.iloc[:,0], x.iloc[:,1])

m,n = x.shape
y = y[:,np.newaxis]
theta = np.zeros((n,1))
lmbda = 1

#j = costFunc(x,y,theta,lmbda)
#print(j)


#using fmin_tnc function to optimize our model
output = opt.fmin_tnc(func = costFunc, x0 = theta.flatten(), fprime = gradient, args = (x, y.flatten(), lmbda))
theta = output[0]
print(theta)

