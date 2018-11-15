import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.io import loadmat

def sigmoid(x):
    return 1/(1+np.exp(-x))
def costFunc(theta, x, y, lmbda):
    m = len(y)
    c1 = np.multiply(y, np.log(sigmoid(x @ theta)))
    c2 = np.multiply((1-y), np.log(1-sigmoid(np.dot(x, theta))))
    c12 = (np.sum(c1 + c2)/(-m)) + (np.sum(theta[1:] ** 2)*(lmbda/(2*m)))
    return c12
def gradientReg(theta,X,y,lmbda):
    m = len(y)
    temp = sigmoid(np.dot(X, theta)) - y
    temp = np.dot(temp.T, X).T / m + theta * lmbda / m
    temp[0] = temp[0] - theta[0] * lmbda / m
    return temp

data = loadmat('ex3data1.mat')
x = data['X']
y = data['y']

# Visualizing the data 

#_, axarr = plt.subplots(10,10,figsize=(10,10))
#for i in range(10):
#    for j in range(10):
#       axarr[i,j].imshow(x[np.random.randint(x.shape[0])].reshape((20,20), order = 'F'))          
#       axarr[i,j].axis('off')
       
# Adding the intercept term
       
x = np.hstack((np.ones((len(y),1)), x))
(m,n) = x.shape

# Using fmin_cfg function in scipy to find optimal values

lmbda = 0.1
k = 10
theta = np.zeros((k,n))
for i in range(k):
    digit = i if i else 10
    theta[i] = opt.fmin_cg(f = costFunc, x0 = theta[i],  fprime = gradientReg,args = (x, (y == digit).flatten(), lmbda), maxiter = 1000)


pred = np.argmax(x @ theta.T, axis = 1)
pred = [e if e else 10 for e in pred]
print(np.mean(pred == y.flatten()) * 100)
# accuracy = 96.46%