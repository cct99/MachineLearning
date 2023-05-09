### In this assignment, we use the MNIST dataset from Keras so we do not need any files to load dataset. 

# Importing necessary libraries
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
 
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize the data
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Add noise to the data
noise_factor = 0.25
X_train_noisy = X_train + noise_factor * np.random.normal(size=X_train.shape)
X_test_noisy = X_test + noise_factor * np.random.normal(size=X_test.shape)

# Ensure that the noisy data is still normalized
X_train_noisy = np.clip(X_train_noisy, 0., 1.)
X_test_noisy = np.clip(X_test_noisy, 0., 1.)

# Reshape the data
X_train = X_train.reshape(X_train.shape[0], 784)
X_test = X_test.reshape(X_test.shape[0], 784)
X_train_noisy = X_train_noisy.reshape(X_train_noisy.shape[0], 784)
X_test_noisy = X_test_noisy.reshape(X_test_noisy.shape[0], 784)

# Define the dimensions of the input data
input_shape = (784,)

# Define the deep AE model 
input_img = Input(shape=input_shape)                 # Input Layer

encoded1 = Dense(128, activation='relu')(input_img)  # Hidden Encoder Layer 1
encoded2 = Dense(64, activation='relu')(encoded1)    # Hidden Encoder Layer 2
bottleneck = Dense(32, activation='relu')(encoded2)  # Bottleneck Layer
decoded1 = Dense(64, activation='relu')(bottleneck)  # Hidden Decoder Layer 1
decoded2 = Dense(128, activation='relu')(decoded1)   # Hidden Decoder Layer 2

output_img = Dense(784, activation='relu')(decoded2) # Output Layer

# Create the deep AE model
deep_ae = Model(input_img, output_img)

# Compile the model using adam optimizer and MSE loss function
deep_ae.compile(optimizer='adam', loss='mean_squared_error')

# Fit the model using the noisy train data as input and the clean train data as target
deep_ae.fit(X_train_noisy.reshape(-1, 784), X_train.reshape(-1, 784), epochs=50, batch_size=256, shuffle=True, validation_split=0.2)

reconstructed_imgs = deep_ae.predict(X_test)

plt.figure(figsize=(20, 4))
for i in range(10):
    # original
    plt.subplot(2, 10, i + 1)
    plt.imshow(X_test[i].reshape(28, 28))
    plt.gray()
    plt.axis('off')
 
    # reconstruction
    plt.subplot(2, 10, i + 1 + 10)
    plt.imshow(reconstructed_imgs[i].reshape(28, 28))
    plt.gray()
    plt.axis('off')
 
plt.tight_layout()
plt.show()
