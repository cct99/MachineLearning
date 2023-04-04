#Importing libraries
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import linear_model

#set default figure size
mpl.rcParams['figure.figsize'] = (12, 8)

data = pd.read_csv('Lab2_dataset.csv')
data_train = data.loc[0:11,['X', 'y']]
data_test = data.loc[0:20,['Xtest', 'ytest']]
data_val = data.loc[0:20,['Xval', 'yval']]

# Include a column of 1s in X to represent X_0 that will be multiplied by theta_0
X_train = np.c_[np.ones_like(data_train['X']), data_train['X']]
y_train = np.c_[data_train['y']]

X_val = np.c_[np.ones_like(data_val['Xval']), data_val['Xval']]
y_val = np.c_[data_val['yval']]

X_test = np.c_[np.ones_like(data_test['Xtest']), data_test['Xtest']]
y_test = np.c_[data_test['ytest']]

print('X_train:') 
print(X_train)
print('y_train:')
print(y_train)

def plotData(X, y, theta = np.array(([0],[0])), reg = 0):
    plt.figure(figsize=(12, 8))
    plt.scatter(X[:, 1], y, s = 50, c = 'red', marker = 'x', linewidths = 1, label = 'Data')
    plt.grid(True)
    plt.title('Water Flow Data')
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    if theta.any() != 0:
        plt.plot(np.linspace(X.min(), X.max()), theta[0] + theta[1] * np.linspace(X.min(), X.max()), 
                                                 label = 'Optimized linear fit')
        plt.title('Water Data: Linear Fit')
        
    plt.legend()

plotData(X_train, y_train)

def cost(theta, X, y, reg = 0):
    m = y.size
    f = np.dot(X,theta).reshape((m, 1))
    J1 = (1 / (2 * m)) * np.sum(np.square(f - y))
    J2 = (reg / (2 * m)) * theta[1:].T.dot(theta[1:])
    J = J1 + J2
    grad = ((1 / m)*(X.T.dot(f - y)) + (reg / m) * np.r_[[[0]], theta[1:].reshape(-1, 1)]).ravel() 
    return J, grad

def gradient_descent(alpha, x, y, numIterations):
    m = y.size 
    theta = np.ones(2)
    J, grad = cost(theta, x, y, reg = 0)
    previous_J = J + 1  
    iter = 0
    J_hist = []
    while abs(J - previous_J) > 1e-3 and iter < numIterations:
        previous_J = J
        theta = theta - alpha * grad
        J, grad = cost(theta, x, y, reg = 0)
        J_hist.append(J)
        print(iter, J)
        iter += 1
    plt.plot(range(len(J_hist)), J_hist)
    plt.xlabel('Iteration')
    plt.ylabel('Mean Square Error (J)')
    plt.title('Mean Square Error versus Iteration Number') 
    plt.show()
    return theta

# Trying out different learning rate
alpha = 0.1 
theta = gradient_descent(alpha, X_train, y_train, 10000)
alpha = 0.001 
theta = gradient_descent(alpha, X_train, y_train, 10000)
alpha = 0.0001 
theta = gradient_descent(alpha, X_train, y_train, 10000)

plotData(X_train, y_train, theta)

from scipy.optimize import minimize
def optimalTheta(theta, X, y, reg = 0):
    #Nelder-Mead yields best fit
    res = minimize(fun = cost, x0 = theta, args = (X, y, reg), method = 'Nelder-Mead', jac = True)
    return res.x

initial_theta = np.ones((X_train.shape[1], 1))
opt_theta = optimalTheta(initial_theta, X_train, y_train)
print("Optimized theta: {0}".format(opt_theta))
plotData(X_train, y_train, opt_theta)

#Train our linear regression model
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

theta_lr = np.array([regr.intercept_[0], regr.coef_[0][1]])

# Calculate the predicted values for the model obtained using LinearRegression
y_pred_lr = theta_lr[0] + theta_lr[1] * X_train[:,1]
y_pred_gd = theta[0] + theta[1] * X_train[:,1]
y_pred_opt = opt_theta[0] + opt_theta[1] * X_train[:,1]

# Plot the data and all 3 the models in the same figure
plt.scatter(X_train[:,1], y_train, color = 'red', marker = 'x', label = 'Data')
plt.plot(X_train[:,1], y_pred_lr, color = 'blue', label = 'Linear Regression')
plt.plot(X_train[:,1], y_pred_gd, color = 'green', label = 'Gradient Descent')
plt.plot(X_train[:,1], y_pred_opt, color = 'orange', label = 'Minimization Model')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Models')
plt.legend()
plt.show()

def plotLearningCurve(theta, X, y, Xval, yval, reg = 0):
    m = y.size
    
    error_train = np.zeros((m, 1))
    error_val = np.zeros((m, 1))
    
    example_num = np.arange(1, (X.shape[0] + 1))
    for i in np.arange(m):
        
        opt_theta = optimalTheta(theta, X[:i + 1], y[:i + 1], reg)
        error_train[i] = cost(opt_theta, X[:i + 1], y[:i + 1], reg)[0]
        error_val[i] = cost(opt_theta, Xval, yval, reg)[0]
    
    plt.figure(figsize = (12, 8))
    plt.plot(example_num, error_train, label = 'Training Error')
    plt.plot(example_num, error_val, label = 'Validation Error')
    plt.title('Learning Curve: No Regularization')
    if reg != 0:
        plt.title('Learning Curve: Lambda = {0}'.format(reg))
    plt.xlabel('Number of training examples')
    plt.ylabel('Mean Square Error')
    plt.legend()
    plt.grid(True)

initial_theta = np.ones((X_train.shape[1], 1))
plotLearningCurve(initial_theta, X_train, y_train, X_val, y_val)

def create_higher_order_features(X_higher_order, p):
    # create higher order features
    for i in np.arange(p):
        dimension = i + 2
        X_higher_order = np.insert(X_higher_order, X_higher_order.shape[1], np.power(X_higher_order[:,1], dimension), axis = 1)
    # perform normalization feature normalization
    normalized_X = X_higher_order
    mean = np.mean(normalized_X, axis = 0)
    normalized_X[:,1:] = normalized_X[:,1:] - mean[1:]
    standard_deviation = np.std(normalized_X, axis = 0)
    normalized_X[:,1:] = normalized_X[:,1:] / standard_deviation[1:]
    
    return X_higher_order, normalized_X

#Polynomial Regression function that creates higher order features, scale them and find optimal theta
def polynomial_regression(X, y, degree, num_points, reg = 0):
    X_higher_order = create_higher_order_features(X, degree)[1]
    init_theta = np.ones((X_higher_order.shape[1], 1))
    opt_theta = optimalTheta(init_theta, X_higher_order, y, reg)
    range = np.linspace(-55,50, num_points)
    range_polynomial = np.ones((num_points, 1))
    range_polynomial = np.insert(range_polynomial, range_polynomial.shape[1], range.T, axis = 1)
    range_polynomial = create_higher_order_features(range_polynomial, len(init_theta)-2)[0]
    range_y = range_polynomial @ opt_theta
    plotData(X, y)
    plt.plot(range, range_y, color="blue", linestyle="--", label = 'Optimized Polynomial Fit')
    plt.grid(True)
    if reg != 0:
        plt.title('Polynomial Fit: Regularization of {0}'.format(reg))    
    plt.title('Water Flow Data')
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.legend()
  
polynomial_regression(X_train, y_train, 8, 100)

Xpolynomial = create_higher_order_features(X_train, 8)[1]
Xpolyval = create_higher_order_features(X_val, 8)[0]
initial_theta = np.ones((Xpolynomial.shape[1], 1))

plotLearningCurve(initial_theta, Xpolynomial, y_train, Xpolyval, y_val, 0)
plotLearningCurve(initial_theta, Xpolynomial, y_train, Xpolyval, y_val, 1)
plotLearningCurve(initial_theta, Xpolynomial, y_train, Xpolyval, y_val, 100)
