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
validation_labels=labels[0:VALIDATION_SIZE,:]

train_images=image[VALIDATION_SIZE:,:]
train_label=labels[VALIDATION_SIZE:,:]

print(train_images.shape)

#tensorflow
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
    
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
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

h_conv1 = tf.nn.relu(conv2d(image,W_conv1)+b_conv1)
h_pool1 = max_pool_2xd(h_conv1)


W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2 = max_pool_2xd(h_conv2)

W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

#dropout
keep_prob = tf.placeholder('float')
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

W_fc2 = weight_variable([1024,labels_count])
b_fc2 = bias_variable([labels_count])

y=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

#cost fucntion  ??
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
#evaluation
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))

predict= tf.argmax(y,1)


epoches_completed= 0
index_in_epoch = 0
num_examples = train_images.shape[0]

def next_batch(batch_size):
    global train_images
    global train_label
    global index_in_epoch
    global epoches_completed
    start = index_in_epoch
    index_in_epoch += batch_size
    
    if(index_in_epoch > num_examples):
        epoches_completed +=1
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_label = train_label[perm]
        #start next epoch 
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end],train_label[start:end]
    
#start TensorFlow session
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)

#visualiszaion variables
train_accuracies=[]
validation_accuracies=[]
x_range = []

display_step=1

for i in range(TRAINING_ITERATIONS):
    batch_xs,batch_ys = next_batch(BATCH_SIZE)
    
    if i%display_step ==0 or (i+1) == TRAINING_ITERATIONS:
        train_accuracy = accuracy.eval(feed_dict ={x:batch_xs,y_:batch_ys,keep_prob:1.0})
        if(VALIDATION_SIZE):
            validation_accuracy = accuracy.eval(feed_dict={x:validation_images[0:BATCH_SIZE],y_:validation_labels[0:BATCH_SIZE],keep_prob:1.0})
            print("training acc / validation acc => %.2f /%.2f for step %d"%(train_accuracy,validation_accuracy,i))
            validation_accuracies.append(validation_accuracy)
        else:
            print("training acc => %0.4f for step %d"%(train_accuracy,i))
        train_accuracies.append(train_accuracy)
    x_range.append(i)
    if(i%(display_step*10)==0 & i):
        display_step = display_step*10
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys,keep_prob:DROPOUT})
                                                                                                                                                                                                                                                                    
testdata = pd.read_csv('./input/test.csv')

testimage = testdata.astype(np.float)
testimage = np.multiply(testimage,1.0/255.0)
predicted_label = np.zeros(testimage.shape[0])

predicted_label = predict.eval(feed_dict={x:testimage,keep_prob:DROPOUT})

np.savetxt('submission_softmax.csv', np.c_[range(1,len(testimage)+1),predicted_label], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')

            
    
