#!/usr/bin/env python

# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = "/mnt/raid/projects/udacity/sdc_nd/datasets/traffic-signs-data/train.p"
validation_file= "/mnt/raid/projects/udacity/sdc_nd/datasets/traffic-signs-data/valid.p"
testing_file = "/mnt/raid/projects/udacity/sdc_nd/datasets/traffic-signs-data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results
import numpy as np
import pandas as pd
import csv

# TODO: Number of training examples
n_train = X_train.shape[0]

# TODO: Number of validation examples
n_validation = X_valid.shape[0]

# TODO: Number of testing examples.
n_test = X_test.shape[0]

# TODO: What's the shape of an traffic sign image?
image_shape = X_train.shape[1:]

# TODO: How many unique classes/labels there are in the dataset.
sign_labels = []
labelreader = csv.reader(open('signnames.csv'), delimiter=",")
for next_label in labelreader:
    #print(next_label)
    sign_labels.append(next_label)
sign_labels.pop(0)

n_classes = np.array(sign_labels).shape[0]

print("Number of training examples =", n_train)
print("Number of validation examples = ", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
# Visualizations will be shown in the notebook.
#%matplotlib inline

fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(16,16))

ax[0][0].hist(train['labels'], bins=n_classes)
ax[0][0].set_title('Dist of signs in training set')

ax[0][1].hist(valid['labels'], bins=n_classes)
ax[0][1].set_title('Dist of signs in validation set')

ax[0][2].hist(test['labels'], bins=n_classes)
ax[0][2].set_title('Dist of signs in test set')

ax[1][0].imshow(X_train[0])
ax[1][1].imshow(X_train[1000])
ax[1][2].imshow(X_train[2000])

ax[2][0].imshow(rgb2gray(X_train[0]), cmap='gray')
ax[2][1].imshow(rgb2gray(X_train[1000]), cmap='gray')
ax[2][2].imshow(rgb2gray(X_train[2000]), cmap='gray')

# plt.show()

X_train_gray_normalized = []
X_valid_gray_normalized = []
X_test_gray_normalized = []

### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.
for i in range(n_train):
    X_train_gray_normalized.append(np.reshape((rgb2gray(X_train[i])), [32,32,1]))
#    X_train_gray_normalized.append(np.reshape((rgb2gray(X_train[i])-128)/128, [32,32,1]))

for i in range(n_validation):
    X_valid_gray_normalized.append(np.reshape((rgb2gray(X_valid[i])), [32,32,1]))
#    X_valid_gray_normalized.append(np.reshape((rgb2gray(X_valid[i])-128)/128, [32,32,1]))

for i in range(n_test):
    X_test_gray_normalized.append(np.reshape((rgb2gray(X_test[i])), [32,32,1]))
#    X_test_gray_normalized.append(np.reshape((rgb2gray(X_test[i])-128)/128, [32,32,1]))

print(X_train[0].shape)
print(X_train_gray_normalized[0].shape)
print(X_train[0][12])
print(X_train_gray_normalized[0][12])


X_train_gray_normalized = np.array(X_train_gray_normalized)
X_valid_gray_normalized = np.array(X_valid_gray_normalized)
X_test_gray_normalized = np.array(X_test_gray_normalized)

X_train_normalized = []
X_valid_normalized = []
X_test_normalized = []

for i in range(n_train):
    X_train_normalized.append((X_train[i]-128)/128)

for i in range(n_validation):
    X_valid_normalized.append((X_valid[i]-128)/128)

for i in range(n_test):
    X_test_normalized.append((X_test[i]-128)/128)

X_train_normalized = np.array(X_train_normalized)
X_valid_normalized = np.array(X_valid_normalized)
X_test_normalized = np.array(X_test_normalized)

# print(X_train_gray_normalized.shape)
# print(X_train_normalized.shape)

### Define your architecture here.
### Feel free to use as many code cells as needed.
import tensorflow as tf

def TrafficSignNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    l1_weights = tf.Variable(tf.truncated_normal([5,5,3,18], mu, sigma))
    l1_bias = tf.Variable(tf.zeros(18))
#    l1_weights = tf.Variable(tf.truncated_normal([5,5,1,6], mu, sigma))
#    l1_bias = tf.Variable(tf.zeros(6))
    l1_strides = [1,1,1,1]
    l1_conv = tf.nn.conv2d(x, l1_weights, l1_strides, padding="VALID")
    l1_conv = tf.nn.bias_add(l1_conv, l1_bias)

    # TODO: Activation.
    l1_conv = tf.nn.relu(l1_conv)

    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    l1_ksize = [1,2,2,1]
    l1_pool_strides = [1,2,2,1]
    l1_conv = tf.nn.max_pool(l1_conv,l1_ksize,l1_pool_strides,"VALID")

    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    l2_weights = tf.Variable(tf.truncated_normal([5,5,18,48], mu, sigma))
#    l2_weights = tf.Variable(tf.truncated_normal([5,5,6,16], mu, sigma))
    l2_bias = tf.Variable(tf.zeros(48))
    l2_strides = [1,1,1,1]
    l2_conv = tf.nn.conv2d(l1_conv, l2_weights, l2_strides, padding="VALID")
    l2_conv = tf.nn.bias_add(l2_conv, l2_bias)
    
    # TODO: Activation.
    l2_conv = tf.nn.relu(l2_conv)

    l2_1_1_weights = tf.Variable(tf.truncated_normal([1,1,48,16], mu, sigma))
    l2_1_1_bias = tf.Variable(tf.zeros(16))
    l2_1_1_strides = [1,1,1,1]
    l2_1_1_conv = tf.nn.conv2d(l2_conv,l2_1_1_weights, l2_1_1_strides, padding="VALID")
    l2_1_1_conv = tf.nn.bias_add(l2_1_1_conv, l2_1_1_bias)
    l2_1_1_conv = tf.nn.relu(l2_1_1_conv)

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    l2_ksize = [1,2,2,1]
    l2_pool_strides = [1,2,2,1]
    l2_conv = tf.nn.max_pool(l2_1_1_conv,l2_ksize,l2_pool_strides,"VALID")

    # TODO: Flatten. Input = 5x5x16. Output = 400.
    l2_flat = tf.reshape(l2_conv, [-1,400])
    
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    l3_dense = tf.layers.dense(l2_flat, 300)
    
    # TODO: Activation.
    l3_dense = tf.nn.relu(l3_dense)

    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    l4_dense = tf.layers.dense(l3_dense, 200)
    
    # TODO: Activation.
    l4_dense = tf.nn.relu(l4_dense)

    # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
    logits = tf.layers.dense(l4_dense, 43)
    
    return logits

def TrafficSignGrayNet(x, keep_prob):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    l1_weights = tf.Variable(tf.truncated_normal([5,5,1,32], mu, sigma))
    l1_bias = tf.Variable(tf.zeros(32))
    l1_strides = [1,1,1,1]
    l1_conv = tf.nn.conv2d(x, l1_weights, l1_strides, padding="VALID")
    l1_conv = tf.nn.bias_add(l1_conv, l1_bias)

    # TODO: Activation.
    l1_conv = tf.nn.relu(l1_conv)

    l1_1_1_weights = tf.Variable(tf.truncated_normal([1,1,32,18], mu, sigma))
    l1_1_1_bias = tf.Variable(tf.zeros(18))
    l1_1_1_strides = [1,1,1,1]
    l1_1_1_conv = tf.nn.conv2d(l1_conv,l1_1_1_weights, l1_1_1_strides, padding="VALID")
    l1_1_1_conv = tf.nn.bias_add(l1_1_1_conv, l1_1_1_bias)
    l1_1_1_conv = tf.nn.relu(l1_1_1_conv)

    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    l1_ksize = [1,2,2,1]
    l1_pool_strides = [1,2,2,1]
    l1_conv = tf.nn.max_pool(l1_1_1_conv,l1_ksize,l1_pool_strides,"VALID")

    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    l2_weights = tf.Variable(tf.truncated_normal([5,5,18,48], mu, sigma))
    l2_bias = tf.Variable(tf.zeros(48))
    l2_strides = [1,1,1,1]
    l2_conv = tf.nn.conv2d(l1_conv, l2_weights, l2_strides, padding="VALID")
    l2_conv = tf.nn.bias_add(l2_conv, l2_bias)
    
    # TODO: Activation.
    l2_conv = tf.nn.relu(l2_conv)

    l2_1_1_weights = tf.Variable(tf.truncated_normal([1,1,48,16], mu, sigma))
    l2_1_1_bias = tf.Variable(tf.zeros(16))
    l2_1_1_strides = [1,1,1,1]
    l2_1_1_conv = tf.nn.conv2d(l2_conv,l2_1_1_weights, l2_1_1_strides, padding="VALID")
    l2_1_1_conv = tf.nn.bias_add(l2_1_1_conv, l2_1_1_bias)
    l2_1_1_conv = tf.nn.relu(l2_1_1_conv)

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    l2_ksize = [1,2,2,1]
    l2_pool_strides = [1,2,2,1]
    l2_conv = tf.nn.max_pool(l2_1_1_conv,l2_ksize,l2_pool_strides,"VALID")

    # TODO: Flatten. Input = 5x5x16. Output = 400.
    l2_flat = tf.reshape(l2_conv, [-1,400])
    
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    l3_dense = tf.layers.dense(l2_flat, 300)
    
    # TODO: Activation.
    l3_dense = tf.nn.relu(l3_dense)
    l3_dense = tf.nn.dropout(l3_dense, keep_prob)

    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    l4_dense = tf.layers.dense(l3_dense, 200)
    
    # TODO: Activation.
    l4_dense = tf.nn.relu(l4_dense)
    l4_dense = tf.nn.dropout(l4_dense, keep_prob)

    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    l5_dense = tf.layers.dense(l4_dense, 128)
    
    # TODO: Activation.
    l5_dense = tf.nn.relu(l5_dense)
    l5_dense = tf.nn.dropout(l5_dense, keep_prob)

    # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
    logits = tf.layers.dense(l5_dense, 43)
    
    return logits

### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.
from sklearn.utils import shuffle

use_gray = True

x = None
if use_gray:
    x = tf.placeholder(tf.float32, (None, 32,32,1), name="INPUT")
else:
    x = tf.placeholder(tf.float32, (None, 32,32,3), name="INPUT")
y = tf.placeholder(tf.uint8, (None), name="LABELS")
one_hot_y = tf.one_hot(y,n_classes)
keep_prob = tf.placeholder(tf.float32)

EPOCHS = 30
BATCH_SIZE = 128
rate = 0.001

logits = None
if use_gray:
    logits = TrafficSignGrayNet(x, keep_prob)
else: 
    logits = TrafficSignNet(x, keep_prob)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = n_train
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        if use_gray:
            X_train_gray_normalized, y_train = shuffle(X_train_gray_normalized, y_train)
        else:
            X_train_normalized, y_train = shuffle(X_train_normalized, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x = None
            batch_y = None
            if use_gray:
                batch_x, batch_y = X_train_gray_normalized[offset:end], y_train[offset:end]
            else:
                batch_x, batch_y = X_train_normalized[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.4})

        valid_accuracy = None
        if use_gray:    
            validation_accuracy = evaluate(X_valid_gray_normalized, y_valid)
        else:
            validation_accuracy = evaluate(X_valid_normalized, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './traffic_sign')
    print("Model saved")
