'''
Created on Jan 14, 2017

@author: lukasz
'''
from test.support import TEST_DATA_DIR


#from settings import filepaths as fp
dogs_filepath = '/home/lukasz/Downloads/train/'

from PIL import Image as im
import numpy as np
import os
from resizeimage import resizeimage as ri
import random as random
import tensorflow as tf

def read_jpg(filepath, weight, height, name):
    image = im.open(filepath)
    image = ri.resize_contain(image, [weight,height])
    image=image.convert('L')
    image.save('/home/lukasz/Downloads/tmp/'+name)
    image = np.array(image)
    return image

def read_all_jpgs(dir_path, weight, height, proportion_to_read_in, test_ratio):
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    i = 1
    for filename in os.listdir(dir_path):
        rand = random.random()
        if rand > 1-proportion_to_read_in :
            i+=1
            rand_test_train = random.random()
            if rand_test_train > test_ratio:
                train_labels.append(1 if 'cat' in filename else 0)
                train_data.append(read_jpg(dir_path+filename, weight, height, filename))
            else:
                test_labels.append(1 if 'cat' in filename else 0)
                test_data.append(read_jpg(dir_path+filename, weight, height, filename))
            if i%1000 == 0:
                break
                print(i)
    return np.array(train_data), np.array(train_labels), np.array(test_data), np.array(test_labels)

def np_array_to_vector(A):
    return np.squeeze(np.array([[x.reshape(x.size)] for x in A]))

def labels_to_versor(A, versor_size):
    return np.array([[1 if y == x else 0 for y in range(versor_size)] for x in A])

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

def select_random_batch(a, b, batch_size):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a[:batch_size], shuffled_b[:batch_size]

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    print('before conv',x.get_shape())
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    print('after conv',x.get_shape())
    x = tf.nn.bias_add(x, b)
    print('after adding bias', x.get_shape())
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout, size):
    # Reshape input picture
    color_channels = 1
    x = tf.reshape(x, shape=[-1, size, size, color_channels])
    print('after reshape',x.get_shape())
    
    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    print('conv1',x.get_shape())
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)
    print('conv1 after maxpool',x.get_shape())

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    print('conv2',x.get_shape())
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)
    print('conv2 after conv2d',x.get_shape())

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

if __name__ == '__main__':
    size = 64 #80 #64
    pets = read_all_jpgs(dogs_filepath, size, size, 1, 0.2)
    # pets[0] - imgs
    # pets[1] - labels
    train_data = pets[0]
    train_labels = pets[1]
    test_data = pets[2]
    test_labels = pets[3]
    
    # reshape to vector
    train_data = np_array_to_vector(train_data)
    test_data = np_array_to_vector(test_data)
    train_labels = labels_to_versor(train_labels, 2)
    test_labels = labels_to_versor(test_labels, 2)
    
    #train_data_shuffled, train_labels_shuffled = shuffle_in_unison(train_data, train_labels)
    #test_data_shuffled, test_labels_shuffled = shuffle_in_unison(test_data, test_labels)
    
    # Parameters
    learning_rate = 0.005
    training_iters = 10000
    batch_size = 128
    display_step = 10
    regularization_constant = 0.001  # Choose an appropriate one.
    
    # Network Parameters
    n_input = size*size # data input (img shape: size x size)
    n_classes = 2 # total classes (cat or dog)
    dropout = 0.75 # Dropout, probability to keep units
    
    filter_size = 4
    
    # tf Graph input
    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])
    keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
    
    # Store layers weight & bias
    weights = {
        # 5x5 conv, 1 input, 32 outputs
        'wc1': tf.Variable(tf.random_normal([filter_size, filter_size, 1, 32])),
        # 5x5 conv, 32 inputs, 64 outputs
        'wc2': tf.Variable(tf.random_normal([filter_size, filter_size, 32, 64])),
        # fully connected, 7*7*64 inputs, 1024 outputs
        ## after 2 maxpools image is of 25x25 size
        'wd1': tf.Variable(tf.random_normal([int(size/2/2*size/2/2*64), 1024])),
        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([1024, n_classes]))
    }
    
    biases = {
        'bc1': tf.Variable(tf.random_normal([32])),
        'bc2': tf.Variable(tf.random_normal([64])),
        'bd1': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
        
        
        # Construct model
    pred = conv_net(x, weights, biases, keep_prob, size)
    
    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    #cost = cost + regularization_constant * sum(regularization_losses)
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    # Initializing the variables
    init = tf.global_variables_initializer()
    
    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        step = 1
        # Keep training until reach max iterations
        while step * batch_size < training_iters:
            batch_x, batch_y = select_random_batch(train_data, train_labels, batch_size)
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                           keep_prob: dropout})
            if step % display_step == 0:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                                  y: batch_y,
                                                                  keep_prob: 1.})
                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
            step += 1
        print("Optimization Finished!")
    
        # Calculate accuracy for 256 mnist test images
        print("Testing Accuracy:", \
            sess.run(accuracy, feed_dict={x: test_data,
                                          y: test_labels,
    keep_prob: 1.}))