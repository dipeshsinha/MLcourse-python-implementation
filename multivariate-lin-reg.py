import numpy as np
import pandas as pd

def costFunc(x,y,m,theta):
    temp = np.dot(x,theta) - y
    return np.sum(np.power(temp,2))/(2*m)

def gradient(x,y,theta,alpha,num_iter):
    m = len(y)
    for _ in range(num_iter):
        temp = np.dot(x,theta) - y
        temp = np.dot(x.T,temp)
        theta = theta - (alpha/m)*temp
    return theta
    
def gradient2(X, y, theta, alpha, iterations):
    m = len(y)
    for _ in range(iterations):
        temp = np.dot(X, theta) - y
        temp = np.dot(X.T, temp)
        theta = theta - (alpha/m) * temp
    return theta

data = pd.read_csv('ex1data2.txt',sep=',', header = None)
x = data.iloc[:,0:2]
y = data.iloc[:,2]
m = len(y)

x = (x - np.mean(x))/np.std(x)

o = np.ones((m,1))
x = np.hstack((o,x))
alpha = 0.01
num_iter = 400
theta = np.zeros((3,1))
y = y[:,np.newaxis]
print(theta)
theta = gradient(x,y,theta,alpha,num_iter)

cost = costFunc(x,y,m,theta)
print(theta)