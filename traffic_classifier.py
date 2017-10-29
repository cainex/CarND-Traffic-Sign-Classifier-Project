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

import cv2

def convert_to_gray_and_normalize(x):
    return np.reshape((cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)-[128])/[128], [32,32,1])  

### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.
for i in range(n_train):
    X_train_gray_normalized.append(convert_to_gray_and_normalize(X_train[i]))

for i in range(n_validation):
    X_valid_gray_normalized.append(convert_to_gray_and_normalize(X_valid[i]))

for i in range(n_test):
    X_test_gray_normalized.append(convert_to_gray_and_normalize(X_test[i]))

X_train_gray_normalized = np.array(X_train_gray_normalized)
X_valid_gray_normalized = np.array(X_valid_gray_normalized)
X_test_gray_normalized = np.array(X_test_gray_normalized)


#print("RGB shape ", X_train[0].shape)
#print("GRAY shape ",X_train_gray.shape)
#print(X_train[0][0])
#print(X_train_gray_normalized[0][0])

### Define your architecture here.
### Feel free to use as many code cells as needed.
import tensorflow as tf

def conv_layer(x, filter_shape, input_depth, filter_depth, pad="VALID", name="conv"):
    mu = 0
    sigma = 0.1
    
    l_weights = tf.Variable(tf.truncated_normal([filter_shape[0],filter_shape[1],input_depth,filter_depth], mu, sigma))
    l_bias = tf.Variable(tf.zeros(filter_depth))
    l_strides = [1,1,1,1]
    l_conv = tf.nn.conv2d(x, l_weights, l_strides, padding=pad, name=name)
    l_conv = tf.nn.bias_add(l_conv, l_bias)

    # Activation.
    l_conv = tf.nn.relu(l_conv)

    return l_conv

def inception_layer(x, input_depth, inception_kernels, name="incep"):
    print("Inception layer:: input_depth:", input_depth, " kernels:", inception_kernels, " output_depth:", 4*inception_kernels)
    l_inc_11_conv = conv_layer(x, (1,1), input_depth, inception_kernels, pad="SAME", name=name+"_11")

    l_inc_55_11_conv = conv_layer(x, (1,1), input_depth, inception_kernels, pad="SAME", name=name+"_55_11")
    l_inc_55_conv = conv_layer(l_inc_55_11_conv, (5,5), inception_kernels, inception_kernels, pad="SAME", name=name+"_55")
 
    l_inc_33_11_conv = conv_layer(x, (1,1), input_depth, inception_kernels, pad="SAME", name=name+"_33_11")
    l_inc_33_conv = conv_layer(l_inc_33_11_conv, (3,3), inception_kernels, inception_kernels, pad="SAME", name=name+"_33")

    l_inc_avg_pool_conv = tf.nn.avg_pool(x, [1,2,2,1], [1,1,1,1],"SAME", name=name+"_avgpool")
    l_inc_avg_pool_11_conv = conv_layer(l_inc_avg_pool_conv, (1,1), input_depth, inception_kernels, pad="SAME", name=name+"_avgpool_11")

    l_inc_conv = tf.concat([l_inc_11_conv, l_inc_55_conv, l_inc_33_conv, l_inc_avg_pool_11_conv], axis=3)

    return l_inc_conv
    
def TrafficSignNet(x, keep_prob):    
    h_params = {'l1_kernels': 32,
                'l1_maxpool_ksize': [1,2,2,1],
                'l1_maxpool_strides': [1,2,2,1],
                'l2_kernels': 64,
                'l3_kernels': 8,
                'l4_kernels': 16,
                'l4_maxpool_ksize': [1,2,2,1],
                'l4_maxpool_strides': [1,2,2,1],
                'l5_avgpool_ksize': [1,5,5,1],
                'l5_avgpool_strides': [1,1,1,1]}

    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    # Layer 1 input: 32x32x1  output: 14x14x['l1_kernels']
    l1_conv = conv_layer(x, (5,5), 1, h_params['l1_kernels'], pad="SAME", name="layer1")
    l1_conv = conv_layer(l1_conv, (5,5), h_params['l1_kernels'], h_params['l1_kernels'], name="layer1_a")

    l1_conv = tf.nn.max_pool(l1_conv,h_params['l1_maxpool_ksize'],h_params['l1_maxpool_strides'],"VALID", name="layer1_maxpool")

   # print(tf.size(l1_conv))

    # Layer 2 input: 14x14x['l1_kernels']  output: 10x10x['l2_kernels']
    l2_conv = conv_layer(l1_conv, (5,5), h_params['l1_kernels'], h_params['l2_kernels'], pad="SAME", name="layer2")
    l2_conv = conv_layer(l2_conv, (5,5), h_params['l2_kernels'], h_params['l2_kernels'], name="layer2_a")

    # Layer 3 Inception Layer input: 10x10x['l2_kernels']  output: 10x10x4*['l3_kernels']
    l3_inc_conv = inception_layer(l2_conv, h_params['l2_kernels'], h_params['l3_kernels'], name="layer3")

    # Layer 4 Inception Layer input: 10x10x4*['l3_kernels'] output: 5x5x4*['l4_kernels']
    l4_inc_conv = inception_layer(l3_inc_conv, 4*h_params['l3_kernels'], h_params['l4_kernels'], name="layer4")
    l4_inc_conv = tf.nn.max_pool(l4_inc_conv, h_params['l4_maxpool_ksize'], h_params['l4_maxpool_strides'],"VALID", name="layer4_maxpool")

    # Layer 5 Global Average Pooling input: 5x5x4*['l4_kernels'] output: 1x1x4*['l4_kernels']
    l5_avgpool = tf.nn.avg_pool(l4_inc_conv, h_params['l5_avgpool_ksize'], h_params['l5_avgpool_strides'],"VALID", name="layer5_avgpool")
    l5_avgpool = tf.nn.dropout(l5_avgpool, keep_prob)

    l5_avgpool_flat = tf.reshape(l5_avgpool, [-1, 4*h_params['l4_kernels']])
    # Layer 6 Dense layer to get final logits
    logits = tf.layers.dense(l5_avgpool_flat, 43)
    
    return logits

def TrafficSignGrayNet(x, keep_prob):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    # Layer 1
    l1_conv = conv_layer(x, (5,5), 1, 32)
    l1_1_1_conv = conv_layer(l1_conv, (1,1), 32, 18)

    l1_ksize = [1,2,2,1]
    l1_pool_strides = [1,2,2,1]
    l1_1_1_conv = tf.nn.max_pool(l1_1_1_conv,l1_ksize,l1_pool_strides,"VALID")

    # Layer 2
    l2_conv = conv_layer(l1_1_1_conv, (5,5), 18, 48)
    l2_1_1_conv = conv_layer(l2_conv, (1,1), 48, 16)

    l2_ksize = [1,2,2,1]
    l2_pool_strides = [1,2,2,1]
    l2_1_1_conv = tf.nn.max_pool(l2_1_1_conv,l2_ksize,l2_pool_strides,"VALID")

    l2_flat = tf.reshape(l2_1_1_conv, [-1,400])
    
    # Layer 3
    l3_dense = tf.layers.dense(l2_flat, 300)
    l3_dense = tf.nn.relu(l3_dense)
    l3_dense = tf.nn.dropout(l3_dense, keep_prob)

    # Layer 4
    l4_dense = tf.layers.dense(l3_dense, 200)
    l4_dense = tf.nn.relu(l4_dense)
    l4_dense = tf.nn.dropout(l4_dense, keep_prob)

    # Layer 5
    l5_dense = tf.layers.dense(l4_dense, 128)
    l5_dense = tf.nn.relu(l5_dense)
    l5_dense = tf.nn.dropout(l5_dense, keep_prob)

    # Output Layer
    logits = tf.layers.dense(l5_dense, 43)
    
    return logits

### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.
from sklearn.utils import shuffle


x = tf.placeholder(tf.float32, (None, 32,32,1), name="INPUT")
y = tf.placeholder(tf.uint8, (None), name="LABELS")
one_hot_y = tf.one_hot(y,n_classes)
keep_prob = tf.placeholder(tf.float32)

EPOCHS = 30
BATCH_SIZE = 128
rate = 0.001

#logits = TrafficSignGrayNet(x, keep_prob)
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
        X_train_gray_normalized, y_train = shuffle(X_train_gray_normalized, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train_gray_normalized[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.4})

        validation_accuracy = evaluate(X_valid_gray_normalized, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './traffic_sign')
    print("Model saved")

### Test model
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test_gray_normalized, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))


def import_and_normalize(filename):
    image = cv2.imread(filename)
    return convert_to_gray_and_normalize(image)

my_image_X = []
my_image_Y = []

my_image_X.append(import_and_normalize("images/general_caution/img1.jpg"))
my_image_Y.append(18)
my_image_X.append(import_and_normalize("images/general_caution/img2.jpg"))
my_image_Y.append(18)
my_image_X.append(import_and_normalize("images/no_entry/img1.jpg"))
my_image_Y.append(17)
my_image_X.append(import_and_normalize("images/speed_limit_30kmh/img1.jpg"))
my_image_Y.append(1)
my_image_X.append(import_and_normalize("images/speed_limit_30kmh/img2.jpg"))
my_image_Y.append(1)
my_image_X.append(import_and_normalize("images/speed_limit_70kmh/img1.png"))
my_image_Y.append(4)
my_image_X.append(import_and_normalize("images/speed_limit_70kmh/img2.jpg"))
my_image_Y.append(4)
my_image_X.append(import_and_normalize("images/turn_right_ahead/img1.jpg"))
my_image_Y.append(33)

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    my_image_prediction = sess.run(tf.argmax(logits, 1), feed_dict={x:my_image_X, keep_prob: 1.0})
    print("My image prediction = ", my_image_prediction)

