#Import necessary libraries.
import numpy as np
from numpy import unique
from numpy import where
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Here we use loadmat to process the dataset. 
# Since it returns a dict, we use X=data['X'] so that we can access the features of the dataset.
data = loadmat('Lab5_Data.mat') # Load the data 
X = data['X']


print("Dimensions of the data: ", X.shape) # Obtain the dimension of the dataset


plt.scatter(X[:, 0], X[:, 1]) # Plot the scatter plot
plt.show()

# In this step, we define the K-Means model and fit the model with our dataset, X.
model = KMeans(n_clusters=2) # define the model
model.fit(X) # fit the model

# Then, use the model for clustering and provide a scatter plot of the clustering result. 
# As shown in the figure, the '+' symbol shows the centroid.
yhat = model.predict(X) # assign a cluster to each example

clusters = unique(yhat) # retrieve unique clusters

for cluster in clusters:
    row_ix = where(yhat == cluster) # get row indexes for samples with this cluster
    plt.scatter(X[row_ix, 0], X[row_ix, 1]) # create scatter of these samples
    
# plot the centroids
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], marker='+', s=100, c='k', linewidth=2)
plt.xlabel('Exam-1 Score')
plt.ylabel('Exam-2 Score')

plt.show() # show the plot

# Here we attempt to obtain the optimal K by using the Elbow Method. The optimal K can be determined by the visualizer. 
# It is positioned at k = 3 (the dotted vertical line).
from yellowbrick.cluster import KElbowVisualizer

# create the visualizer with K range (2,7)
model = KMeans()
visualizer = KElbowVisualizer(model, k=(2,7))

# fit the data to the visualizer
visualizer.fit(X) 

# finalize and render the figure
visualizer.show()

# define the model with optimal K
model = KMeans(n_clusters=3)

# fit the model to the data
model.fit(X)

# assign a cluster to each example
yhat = model.predict(X)

# retrieve unique clusters
clusters = unique(yhat)

# plot the clusters and centroids
for cluster in clusters:
    # get row indexes for samples with this cluster
    row_ix = where(yhat == cluster)
    # create scatter of these samples
    plt.scatter(X[row_ix, 0], X[row_ix, 1])
    
# plot the centroids
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], marker='+', s=100, c='k', linewidth=2)
plt.xlabel('Exam-1 Score')
plt.ylabel('Exam-2 Score')
# show the plot
plt.show()
