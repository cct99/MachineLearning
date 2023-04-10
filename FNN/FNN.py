### In this assignment, we use the MNIST dataset from Keras so we do not need any files to load dataset. 

# Import the Libraries
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from keras.wrappers.scikit_learn import KerasClassifier

# Other machine learning modules
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Numpy and Pandas
import numpy as np
import pandas

# Importing additional libraries
from matplotlib import pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.utils import np_utils
from sklearn.metrics import accuracy_score

(X_train, y_train), (X_test, y_test) = mnist.load_data() # Load the training data

# Obtain the dimensions and the number of samples
num_train_samples, img_rows, img_cols = X_train.shape
num_test_samples, img_rows, img_cols = X_test.shape

# Reshape the train and test features
X_train = X_train.reshape(X_train.shape[0], 784)
X_test = X_test.reshape(X_test.shape[0], 784)

# Feature scaling
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# Find the number of train samples and their dimension
print("Number of training samples:", num_train_samples)
print("Training data dimension:", (img_rows, img_cols))

# Find the number of test samples and their dimension
print("Number of testing samples:", num_test_samples)
print("Testing data dimension:", (img_rows, img_cols))
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

for i in range(9): # Construct the first 9 samples of the dataset
 plt.subplot(330 + 1 + i)
 plt.imshow(X_train[i].reshape((img_rows, img_cols)), cmap=plt.get_cmap('gray'))
# show the figure
plt.show()

# Define model function
def tuned_model(neurons=1, dropout_rate=0.0, layers=2, act_h='relu', act_o='softmax'):
    # create model
    model = Sequential()
    model.add(Dense(neurons, input_dim=28*28, kernel_initializer='normal', activation=act_h))
    
    # Add hidden layers based on the number specified in function argument
    if layers == 2:
        model.add(Dense(neurons, kernel_initializer='normal', activation=act_h))
    elif layers == 3:
        model.add(Dense(neurons, kernel_initializer='normal', activation=act_h))
        model.add(Dropout(dropout_rate))
        model.add(Dense(neurons, kernel_initializer='normal', activation=act_h))
    elif layers == 4:
        model.add(Dense(neurons, kernel_initializer='normal', activation=act_h))
        model.add(Dropout(dropout_rate))
        model.add(Dense(neurons, kernel_initializer='normal', activation=act_h))
        model.add(Dropout(dropout_rate))
        model.add(Dense(neurons, kernel_initializer='normal', activation=act_h))
    
    # add output layer
    model.add(Dense(10, kernel_initializer='normal', activation=act_o))
   
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Create Model 
from sklearn.model_selection import RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
classifier = KerasClassifier(build_fn=tuned_model, epochs=50, batch_size=200, verbose=0)

# Do Grid Search using RandomizedSearchCV
layers = [2, 3, 4] # Defining search space
neurons = [50, 100]
dropout_rate = [0.0, 0.2]
act_h = ['relu', 'sigmoid']
act_o = ['softmax', 'sigmoid']

param_grid = dict(neurons=neurons, dropout_rate=dropout_rate, layers=layers, act_h=act_h, act_o=act_o)

# Run RandomizedSearchCV
grid = RandomizedSearchCV(estimator=classifier, param_distributions=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X_train, y_train)

# Printing out the grid search result and determine the best hyper-parameter result.
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# Here we train the best model by using the best hyper-parameters that we obtained in the previous section.
best_model = grid_result.best_estimator_.model # Train model with best hyper-parameters
best_model.fit(X_train, y_train, epochs=50, batch_size=200, verbose=0) 

# Create the prediction model
test_predicted = best_model.predict(X_test)

# Do conversion into one-hot-encoding so that we can compare the predicted label to the actual label and report the model's accuracy score.
test_predicted = np.argmax(test_predicted, axis=1) # Convert predicted labels to one-hot-encoding
test_predicted = np_utils.to_categorical(test_predicted)

# Calculate accuracy of the model
accuracy = accuracy_score(y_test, test_predicted)
print("Model accuracy on test set: {:.2f}%".format(accuracy * 100))
