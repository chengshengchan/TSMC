
'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''
# Based on above project, modified by James Chan


#from __future__ import print_function

import tensorflow as tf
from mnist import loader

import numpy as np
import pdb, time

# Import MNIST data
# (mnist data provided by tensorflow)
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# load mnist data manually
def load_mnist_data(flag, data_path='data'):
    data_loader = loader.MNIST(data_path)
    if flag == 'train':
        ims, labels = data_loader.load_training()
    elif flag == 'test':
        ims, labels = data_loader.load_testing()
    else:
        raise ValueError("Error. Load training or testing data.\nUse: load_mnist_data(flag). flag = \'train\' or \'test\'.\n")
    ims = ims/255.0
    ims_mean = np.mean(ims, axis=0)
    return ims, labels, ims_mean

# Parameters
learning_rate = 0.001
#training_iters = 20000
training_epochs = 100
batch_size = 100
display_step = 10
stddev=0.01
# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
#dropout = 0.75 # Dropout, probability to keep units



# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

def lenet():
    # tf Graph input
    x = tf.placeholder(tf.float32, [None, n_input])
    #y = tf.placeholder(tf.float32, [None, n_classes])
    y = tf.placeholder(tf.int32,[None])
    dropout = tf.placeholder(tf.float32) #dropout (keep probability)
    
    # Store layers weight & bias
    weights = {
        # 5x5 conv, 1 input, 32 outputs
        'wc1': tf.Variable(tf.truncated_normal([5, 5, 1, 32], mean=0, stddev=stddev)),
        # 5x5 conv, 32 inputs, 64 outputs
        'wc2': tf.Variable(tf.truncated_normal([5, 5, 32, 64], mean=0, stddev=stddev)),
        # fully connected, 7*7*64 inputs, 1024 outputs
        'wd1': tf.Variable(tf.truncated_normal([7*7*64, 1024], mean=0, stddev=stddev)),
        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.truncated_normal([1024, n_classes], mean=0, stddev=stddev))
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([32])),
        'bc2': tf.Variable(tf.random_normal([64])),
        'bd1': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Construct model
    #pred = conv_net(x, weights, biases, keep_prob)
    x_reshape = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x_reshape, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, keep_prob = 1-dropout)

    # Output, class prediction
    pred = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    # Define loss and optimizer
    #one_hot_y = tf.one_hot(y, n_classes, on_value=1, off_value=0, axis=-1)
    #one_hot_y = tf.cast(one_hot_y, tf.float32)
    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=one_hot_y))

    

    probs = tf.nn.softmax(pred)
    log_probs = tf.log(probs + 1e-8)

    one_hot_y = tf.one_hot(y, n_classes, on_value=1, off_value=0, axis=-1)
    #print one_hot_y.get_shape()
    #cross_entropy_loss = - tf.mul(y,log_probs)
    cross_entropy_loss = - tf.mul(tf.cast(one_hot_y, tf.float32),log_probs)
    
    loss = tf.reduce_sum(cross_entropy_loss)


    return x, y, dropout, loss, pred, one_hot_y


# Initializing the variables
#init = tf.global_variables_initializer()




ims, labels, ims_mean = load_mnist_data('train', data_path='data')
ims_test, labels_test, _ = load_mnist_data('train', data_path='data')

order_list = range(len(ims))

# Launch the graph
with tf.Session() as sess:
    epoch = 0
    iter_per_epoch = len(ims)/batch_size
    step = 0
    # Keep training until reach max iterations
    x, y, dropout, cost, pred, one_hot_y = lenet()
    train_loss = cost/batch_size
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(train_loss)


    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(one_hot_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))




    try:
        init = tf.initialize_all_variables()
    except:
        init = tf.global_variables_initializer()
    sess.run(init)
    # Before Training
    begin = time.time()
    Train_Loss = 0
    Test_Loss = 0
    Train_Acc = 0
    Test_Acc = 0
    for idx in xrange(iter_per_epoch):
        batch_xs = ims[order_list[idx*batch_size:(idx+1)*batch_size]] - ims_mean
        batch_ys = labels[order_list[idx*batch_size:(idx+1)*batch_size]]
        #batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        C, A = sess.run([cost, accuracy], feed_dict={x: batch_xs, y: batch_ys, dropout: 0.0})
        Train_Loss += C/batch_size
        Train_Acc += A
    test_batch=1000
    test_iter = len(ims_test)/test_batch
    for idx in xrange(test_iter):
        batch_xs = ims_test[order_list[idx*test_batch:(idx+1)*test_batch]] - ims_mean
        batch_ys = labels_test[order_list[idx*test_batch:(idx+1)*test_batch]]
        C, A = sess.run([cost, accuracy], feed_dict={x: batch_xs, y: batch_ys, dropout: 0.0})
        Test_Loss += C/test_batch
        Test_Acc += A

    print "Epoch %d, Training: loss=%f, acc=%f.\t\tTesting: loss=%f, acc=%f"%(epoch, Train_Loss/iter_per_epoch, Train_Acc/iter_per_epoch, 
                                                                                     Test_Loss/test_iter, Test_Acc/test_iter)
    duration = time.time()-begin
    print "Cost %f seconds"%(duration)
    print '-----------------init-----------------'
    while epoch < training_epochs:
        begin = time.time()
        Train_Loss = 0
        Test_Loss = 0
        Train_Acc = 0
        Test_Acc = 0
        for idx in xrange(iter_per_epoch):
            batch_xs = ims[order_list[idx*batch_size:(idx+1)*batch_size]] - ims_mean
            batch_ys = labels[order_list[idx*batch_size:(idx+1)*batch_size]]
            #batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop)
            C, A, _ = sess.run([cost, accuracy, optimizer], feed_dict={x: batch_xs, y: batch_ys, dropout: 0.5})
            Train_Loss += C/batch_size
            Train_Acc += A
            if step % display_step == 0:
                pass
                # Calculate batch loss and accuracy
                #loss, acc, OY = sess.run([cost, accuracy, probs], feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.0})
                #print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                #      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                #      "{:.5f}".format(acc))
            step += 1
        test_batch=100
        test_iter = len(ims_test)/test_batch
        for idx in xrange(test_iter):
            batch_xs = ims_test[order_list[idx*test_batch:(idx+1)*test_batch]] - ims_mean
            batch_ys = labels_test[order_list[idx*test_batch:(idx+1)*test_batch]]
            C, A = sess.run([cost, accuracy], feed_dict={x: batch_xs, y: batch_ys, dropout: 0.0})
            Test_Loss += C/test_batch
            Test_Acc += A
        print "Epoch %d, Training: loss=%f, acc=%f.\t\tTesting: loss=%f, acc=%f"%(epoch, Train_Loss/iter_per_epoch, Train_Acc/iter_per_epoch, 
                                                                                         Test_Loss/test_iter, Test_Acc/test_iter)
        duration = time.time()-begin
        print "Cost %f seconds"%(duration)
    print("Optimization Finished!")

