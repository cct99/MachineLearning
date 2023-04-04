#Import libraries
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures

def loaddata(file, delimeter):
    data = np.loadtxt(file, delimiter=delimeter)
    print('Dimensions: ',data.shape)
    print(data[1:6,:])
    return(data)

def plotData(data, label_x, label_y, label_pos, label_neg, axes=None):
    # Get indexes for class 0 and class 1
    neg = data[:,2] == 0
    pos = data[:,2] == 1
    
    # If no specific axes object has been passed, get the current axes.
    if axes == None:
        axes = plt.gca()
    axes.scatter(data[pos][:,0], data[pos][:,1], marker='+', c='k', s=60, linewidth=2, label=label_pos)
    axes.scatter(data[neg][:,0], data[neg][:,1], c='y', s=60, label=label_neg)
    axes.set_xlabel(label_x)
    axes.set_ylabel(label_y)
    axes.legend(frameon= True, fancybox = True);

#Call plotData function to visualize the scatter plot
data = loaddata('Dataset-2.txt', ',')

plotData(data, 'Test 1 score', 'Test 2 score', 'Accepted', 'Rejected')

#Separating X and y which are the features and labels, respectively, by using the numpy array.
X = data[:,0:2]
y = np.c_[data[:,2]]
poly = PolynomialFeatures(6)
X_poly = poly.fit_transform(X)

def sigmoid(z):
    return(1 / (1 + np.exp(-z)))
    
sigmoid(0)

def costFunction(theta, X_poly, y, reg):
    m = y.size
    f = sigmoid(X_poly.dot(theta.reshape(-1, 1)))
    reg_term = (reg/(2*m)) * np.sum(np.square(theta[1:]))
    J = -1*(1/m)*(np.log(f).T.dot(y)+np.log(1-f).T.dot(1-y)) + reg_term
               
    if np.isnan(J[0]):
        return(np.inf)
    return(J[0])
    
def gradient(theta, X_poly, y, reg):
    m = y.size
    f = sigmoid(X_poly.dot(theta.reshape(-1, 1)))
    reg_term = (reg/m) * np.r_[[[0]],theta[1:].reshape(-1,1)]
    grad = (1/m) * X_poly.T.dot(f-y) + reg_term

    return(grad.flatten())
     
initial_theta = np.zeros(X_poly.shape[1])
reg = 0
cost = costFunction(initial_theta, X_poly, y, reg)
grad = gradient(initial_theta, X_poly, y, reg)
print('Cost: \n', cost)
print('Grad: \n', grad)

opt_theta = minimize(costFunction, initial_theta, args=(X_poly,y, reg), method=None, jac=gradient, options={'maxiter':400})
     
print('Thetas: \n', opt_theta)

def classify(theta, X_poly, threshold=0.5):
    p = sigmoid(X_poly.dot(theta.T)) >= threshold
    return(p.astype('int'))
    
p = classify(opt_theta.x, X_poly) 
print('Train accuracy {}%'.format(100*sum(p == y.ravel())/p.size))

plotData(data, 'Test 1 score', 'Test 2 score', 'Accepted', 'Rejected')
x1_min, x1_max = X_poly[:,1].min(), X_poly[:,1].max(),
x2_min, x2_max = X_poly[:,2].min(), X_poly[:,2].max(),
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
f = sigmoid(poly.fit_transform(np.c_[xx1.ravel(), xx2.ravel()]).dot(opt_theta.x))
f = f.reshape(xx1.shape)  
plt.contour(xx1, xx2, f, [0.5], linewidths=1, colors='b')

reg = 1
cost = costFunction(initial_theta, X_poly, y, reg)
grad = gradient(initial_theta, X_poly, y, reg)

opt_theta = minimize(costFunction, initial_theta, args=(X_poly,y, reg), method=None, jac=gradient, options={'maxiter':400})
p = classify(opt_theta.x, X_poly) 
print('Train accuracy {}%'.format(100*sum(p == y.ravel())/p.size))
plotData(data, 'Test 1 score', 'Test 2 score', 'Accepted', 'Rejected')
x1_min, x1_max = X_poly[:,1].min(), X_poly[:,1].max(),
x2_min, x2_max = X_poly[:,2].min(), X_poly[:,2].max(),
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
f = sigmoid(poly.fit_transform(np.c_[xx1.ravel(), xx2.ravel()]).dot(opt_theta.x))
f = f.reshape(xx1.shape)  
plt.contour(xx1, xx2, f, [0.5], linewidths=1, colors='b')

reg = 100
cost = costFunction(initial_theta, X_poly, y, reg)
grad = gradient(initial_theta, X_poly, y, reg)

opt_theta = minimize(costFunction, initial_theta, args=(X_poly,y, reg), method=None, jac=gradient, options={'maxiter':400})
p = classify(opt_theta.x, X_poly) 
print('Train accuracy {}%'.format(100*sum(p == y.ravel())/p.size))
plotData(data, 'Test 1 score', 'Test 2 score', 'Accepted', 'Rejected')
x1_min, x1_max = X_poly[:,1].min(), X_poly[:,1].max(),
x2_min, x2_max = X_poly[:,2].min(), X_poly[:,2].max(),
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
f = sigmoid(poly.fit_transform(np.c_[xx1.ravel(), xx2.ravel()]).dot(opt_theta.x))
f = f.reshape(xx1.shape)  
plt.contour(xx1, xx2, f, [0.5], linewidths=1, colors='b')


from sklearn.linear_model import LogisticRegression
from sklearn import metrics

logisticRegr = LogisticRegression()
logisticRegr.fit(X_poly, y)

predictions = logisticRegr.predict(X_poly)

score = logisticRegr.score(X_poly, y)
print(score)

cm = metrics.confusion_matrix(y, predictions)
print(cm)
