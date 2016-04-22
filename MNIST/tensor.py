import numpy as np
import pandas as pd
import pdb
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import tensorflow as tf


def dense_to_onehot(label_dense,num_class):
    num_labels=label_dense.shape[0]
    offset=np.arange(num_labels)*num_class
    onehot=np.zeros((num_labels,num_class))
    onehot.flat[offset + label_dense.ravel()]=1
    return onehot
    
    
LEARNING_RATE  = 1e-4
TRAINING_ITERATIONS = 2000

DROPOUT = 0.5
BATCH_SIZE = 50

VALIDATION_SIZE=2000
IMAGE_TO_DISPLAY = 10

data = pd.read_csv('./input/train.csv')

#pdb.set_trace()
#so smart I can use the .values attribute to return value
image = data.iloc[:,1:].values
image = image.astype(np.float)
image = np.multiply(image,1.0/255.0)
print(image.shape)

image_size=image.shape[1]
image_width = image_height=np.ceil(np.sqrt(image_size)).astype(np.uint8)

#why?
labels_flat = data[[0]].values.ravel()

labels_count = np.unique(labels_flat).shape[0]

print('label_count =>{0}'.format(labels_count))

#pdb.set_trace()

labels=dense_to_onehot(labels_flat,labels_count)
labels=labels.astype(np.uint8)

validation_images=image[0:VALIDATION_SIZE,:]
validataion_labels=labels[0:VALIDATION_SIZE,:]

train_images=image[VALIDATION_SIZE:,:]
train_label=labels[VALIDATION_SIZE:,:]

print(train_images.shape)

#tensorflow
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
    
def bias_variable(shape):
    inital = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
    
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")

def max_pool_2xd(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

x=tf.placeholder('float',shape=[None,image_size])
y_ = tf.placeholder('float',shape=[None,labels_count])


#I haven't figure the detail of convolution yet,refer to cs231n later
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])

image = tf.reshape(x,[-1,image_width,image_height,1])




