### In this assignment, we use the 'Boston Housing' dataset from Keras so we do not need any files to load dataset. 

import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from numpy import mean
from numpy import cov
from numpy.linalg import eig
from sklearn.decomposition import PCA
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator

import tensorflow as tf
from tensorflow import keras

# Load Dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.boston_housing.load_data()
train_mean = np.mean(X_train, axis=0)
train_std = np.std(X_train, axis=0)
train_features = (X_train - train_mean) / train_std
test_features = (X_test - train_mean) / train_std

# Print the number of features
print("Number of features:", X_train.shape[1])

# Train PCA model
pca = PCA(13)
pca.fit(train_features)

# Print the number of principal components
print(pca.components_.T)
print(pca.explained_variance_ratio_)

# Provide a scree plot and cumulative plot based on the obtained principal components
PC = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12', 'PC13']
plt.bar(PC,pca.explained_variance_ratio_)
plt.xlabel('Principal Components')
plt.ylabel('% Retained Variance')
plt.show()
print(pca.explained_variance_ratio_)

n_comp = np.arange(13)
n_comp = n_comp + 1
cumulative = np.cumsum(pca.explained_variance_ratio_)
ax = figure().gca()
ax.plot(n_comp, cumulative)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel("Principal Components")
plt.ylabel("Cumulative % of Retained Variance")
show()

# Using the cumulative plot, determine the number of principal components to retain at least 80% and 95% of the variance
for i, var in enumerate(cumulative):
    if var >= 0.8:
        print("At least", i+1, "principal components are needed to retain 80% of the variance.")
        break
for i, var in enumerate(cumulative):
    if var >= 0.95:
        print("At least", i+1, "principal components are needed to retain 95% of the variance.")
        break
        
pca2 = PCA(2)
pca2.fit(X_train)
X_P2 = pca2.transform(X_train)

# Print the number of features in the reduced dataset and the retained variance
print("Number of features in the reduced dataset:", X_P2.shape[1])
print("Variance retained:", sum(pca2.explained_variance_ratio_))

# Print the first 5 rows of the reduced dataset
print(X_P2[0:5,:])

# Recover the original data from the reduced data
X_R2 = pca2.inverse_transform(X_P2)

# Print the shape and the first 5 rows of the recovered data
print("Shape of the recovered data:", X_R2.shape)
print(X_R2[0:5,:])    
        
    
