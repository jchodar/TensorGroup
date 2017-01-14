'''
Created on Jan 5, 2017

@author: krzys :)
'''
import tensorflow as tf
import numpy
import pandas
from PIL import Image
import os
from settings.filepaths import filename_data,filename_test, filename_output, visualization_prefix


class Dataset(object):
    def __init__(self, autoshuffle=True):
        self._data = numpy.array([])
        self._labels = numpy.array([])
        self._number_of_samples = 0
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._autoshuffle = True
           
    def get_data(self):
        return self._data
    
    def get_labels(self):
        return self._labels
    
    def get_number_of_samples(self):
        return self._number_of_samples
    
    def get_index_in_epoch(self):
        return self._index_in_epoch
    
    def get_epochs_completed(self):
        return self._epochs_completed
    
    def set_data(self, data):
        self._data = data
        self._number_of_samples = self._data.shape[0]
    
    def set_labels(self, labels):
        self._labels = labels
        
    def reset_epochs(self):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        if self._autoshuffle:
            self.shuffle()
        
    def shuffle(self):
        perm = numpy.arange(self._number_of_samples)
        numpy.random.shuffle(perm)
        self._data = self._data[perm]
        self._labels =self._labels[perm]
        
    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._number_of_samples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            if self._autoshuffle:
                self.shuffle()
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._number_of_samples
        end = self._index_in_epoch
        return self._data[start:end], self._labels[start:end]
    
    def remove_data_front(self, numer_of_samples):
        self._data = self._data[numer_of_samples:]
        self._labels =self._labels[numer_of_samples:]
        self._number_of_samples = self._data.shape[0]
    

class ProgramData(object):
    def __init__(self):
        self.train = Dataset()
        self.test = Dataset(autoshuffle=False)
        self.problem_data = Dataset(autoshuffle=False)
        
    def read_train_file_to_datasets(self, filename, data_transformation_func=lambda x: x,
                                 label_transformation_func=lambda x: x):
        print('reading file: ', filename)
        dataset = pandas.read_csv(filename)
        print('read size:', dataset.size, 'shape', dataset.shape)
        self.train.set_data(data_transformation_func(dataset.iloc[:,1:].values))
        self.train.set_labels(label_transformation_func(dataset.ix[:,:'label'].values))

    def move_part_of_trian_to_test(self, number_of_samples):
        test_data, test_labels = self.train.next_batch(number_of_samples)
        self.test.set_data(test_data)
        self.test.set_labels(test_labels)
        self.train.remove_data_front(number_of_samples)
    
    def read_problem_file_to_dataset(self, filename, data_transformation_func=lambda x: x):
        print('reading file: ', filename)
        dataset = pandas.read_csv(filename)
        print('read size:', dataset.size, 'shape', dataset.shape)
        self.problem_data.set_data(data_transformation_func(dataset.iloc[:,:].values))


def visualise(w_matrix, imagePath):
    if not os.path.exists(imagePath):
        os.mkdir(imagePath)
    
    for i in range(10):
        pic = w_matrix[:,i]
        plus  = numpy.array([(   x if x > 0 else 0) for x in pic])
        plus /= numpy.amax(plus)
        plus *= 255
        plus.resize((28,28))
        minus  = numpy.array([( -x if x < 0 else 0) for x in pic])
        minus /= numpy.amax(minus)
        minus.resize((28,28))
        minus *= 255
        img = Image.fromarray(255-plus)
        img = img.convert("RGB")
        imgpath = imagePath+"r" + str(i) +"p.bmp"
        print("saving img to path:" + imgpath)
        img.save(imgpath)
        img = Image.fromarray(255-minus)
        img = img.convert("RGB")
        imgpath = imagePath+"r" + str(i) +"m.bmp"
        print("saving img to path:" + imgpath)
        img.save(imgpath)        


def labels_to_versor(A, versor_size):
    return numpy.array([[1 if y == x[0] else 0 for y in range(versor_size)] for x in A])

def gray_to_bw(A):
    return numpy.array([[1 if y > 127 else 0 for y in x] for x in A])
    
    
if __name__ == '__main__':
    print('hello')
    
    learning_rate = 0.001
    training_iteration = 5
    batch_size = 100
    display_step = 1
    test_set_size = 1000
    
    process_problem_data = False
    
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])
    tr_rate = tf.placeholder(tf.float32, shape=[])
    
    # Construct a linear model
    _y = tf.matmul(x, W) + b # Softmax
    # Minimize error using cross entropy
    # Cross entropy
    #cost_function = -tf.reduce_sum(y*tf.log(_y))
    cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=_y))
    # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(tr_rate).minimize(cost_function)
    
    if not os.path.exists(filename_data):
        print('Data file not found')
        exit(1)
    
    program_data = ProgramData()
    program_data.read_train_file_to_datasets(filename_data, data_transformation_func=gray_to_bw, label_transformation_func=lambda x: labels_to_versor(x, 10))
    program_data.train.shuffle()
    program_data.move_part_of_trian_to_test(test_set_size)

    # Initializing the variables
    print('init var')
    init = tf.global_variables_initializer()
    
    # Launch the graph
    with tf.Session() as sess:
        print('init session')
        sess.run(init)
    
        # Training cycle
        print('test')
        
        print('start')
        for iteration in range(training_iteration):
            #random.shuffle(aviable_indexes)
            print(str(iteration+1) + ' iteration')
            avg_cost = 0.
            # Loop over all batches
            for i in range(int(program_data.train.get_number_of_samples()/batch_size)):
                rate = learning_rate-learning_rate*(iteration*int(program_data.train.get_number_of_samples()/batch_size) + i )/(training_iteration * int(program_data.train.get_number_of_samples()/batch_size))
                #print(str(i) + ' inner lap')
                # Fit training using batch data
                batch_data, batch_labels = program_data.train.next_batch(batch_size)
                
                sess.run(optimizer, feed_dict={x: batch_data, y: batch_labels, tr_rate : rate})
                # Compute average loss
                avg_cost += sess.run(cost_function, feed_dict={x: batch_data, y: batch_labels})/(int(program_data.train.get_number_of_samples()/batch_size))
            # Display logs per iteration step
            if iteration % display_step == 0:
                print('iteration:', '%04d' % (iteration + 1), 'cost=', '{:.9f}'.format(avg_cost))
                predictions = tf.equal(tf.argmax(_y, 1), tf.argmax(y, 1))
                # Calculate accuracy
                accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
                print('Accuracy:', accuracy.eval({x: program_data.test.get_data(), y: program_data.test.get_labels()}))
                visualise(W.eval(), visualization_prefix + str(iteration) + '/')
    
        print('Tuning completed!')
    
#         # Test the model
#         predictions = tf.equal(tf.argmax(_y, 1), tf.argmax(y, 1))
#         # Calculate accuracy
#         accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
#         print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))
        
        if process_problem_data and os.path.exists(filename_test):
            print('Creating test sets')
            #test_data = readTestFileToDataset(filename_test)
            program_data.read_problem_file_to_dataset(filename_test, lambda x: x)
            print('Test sets created')   
            print('Evaluating test sets')
            #test_data = numpy.array(list(map(lambda x: list(map(lambda y: 1 if int(y) > 128 else 0, x)), test_data)))
            results = tf.argmax(_y, 1).eval({x: program_data.problem_data.get_data()}) 
            print('Test sets evaluated')
            f = open(filename_output, 'w+')
            f.write('ImageId,Label\n')
            for i in range(len(results)):
                f.write(str(i+1) + ';' + str(results[i]) + '\n')
            f.close()
            print('Test sets saved')
    