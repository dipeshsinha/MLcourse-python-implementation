import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def computeCost(x,y,theta):
    temp = np.dot(x,theta) - y
    return (np.sum(np.power(temp,2)))/(2*m)

def gradient(x,y,theta,alpha,iterations):
    for _ in range(iterations):
        temp = np.dot(x,theta) - y
        temp = np.dot(x.T,temp)
        theta= theta - ((alpha/m)*temp)
    return theta

data = pd.read_csv('ex1data1.txt', header = None)
x = data.iloc[:,0]
y = data.iloc[:,1]
m = len(y)

x = x[:,np.newaxis]
y = y[:,np.newaxis]
theta = np.zeros([2,1])
iterations = 2000
alpha = 0.01
ones = np.ones((m,1))
x = np.hstack((ones,x))


theta = gradient(x,y,theta,alpha,iterations)

j = computeCost(x,y,theta)

plt.scatter(x[:,1],y)
plt.plot(x[:,1], np.dot(x,theta),'r')
plt.show()

