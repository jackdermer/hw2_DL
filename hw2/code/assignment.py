from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data, get_batch
from convolution import conv2d

import os
import tensorflow as tf
import numpy as np
import random
import math

class Model(tf.keras.Model):
    def __init__(self):
        """
        This model class will contain the architecture for your CNN that 
        classifies images. Do not modify the constructor, as doing so 
        will break the autograder. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(Model, self).__init__()

        self.batch_size = 100
        self.num_classes = 2
        self.loss_list = [] # Append losses to this list in training so you can visualize loss vs time in main

        # TODO: Initialize all hyperparameters
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        # TODO: Initialize all trainable parameters
        self.C1 = tf.Variable(tf.random.truncated_normal([5,5,3,16], stddev=.1, dtype=tf.float32))
        self.C2 = tf.Variable(tf.random.truncated_normal([5,5,16,20], stddev=.1, dtype=tf.float32))
        self.C3 = tf.Variable(tf.random.truncated_normal([3,3,20,20], stddev=.1, dtype=tf.float32))
        self.W1 = tf.Variable(tf.random.truncated_normal([80,80], stddev=.1, dtype=tf.float32))
        self.b1 = tf.Variable(tf.random.truncated_normal([80], stddev=.1, dtype=tf.float32))
        self.W2 = tf.Variable(tf.random.truncated_normal([80,80], stddev=.1, dtype=tf.float32))
        self.b2 = tf.Variable(tf.random.truncated_normal([80], stddev=.1, dtype=tf.float32))
        self.W3 = tf.Variable(tf.random.truncated_normal([80,2], stddev=.1, dtype=tf.float32))
        self.b3 = tf.Variable(tf.random.truncated_normal([2], stddev=.1, dtype=tf.float32))

    def call(self, inputs, is_testing=False):
        """
        Runs a forward pass on an input batch of images.
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        # Remember that
        # shape of input = (num_inputs (or batch_size), in_height, in_width, in_channels)
        # shape of filter = (filter_height, filter_width, in_channels, out_channels)
        # shape of strides = (batch_stride, height_stride, width_stride, channels_stride)

        x = tf.nn.conv2d(inputs, self.C1, strides=[2,2], padding='SAME')
        mean, variance = tf.nn.moments(x, [0,1,2])
        x = tf.nn.batch_normalization(x, mean=mean, variance=variance, variance_epsilon=1e-5, offset=None, scale=None)
        x = tf.nn.relu(x)
        x = tf.nn.max_pool(x, ksize=[3,3], strides=[2,2], padding='SAME')
        # print(x.shape)

        x = tf.nn.conv2d(x, self.C2, strides=[2,2], padding='SAME')
        mean, variance = tf.nn.moments(x, [0,1,2])
        x = tf.nn.batch_normalization(x, mean=mean, variance=variance, variance_epsilon=1e-5, offset=None, scale=None)
        x = tf.nn.relu(x)
        x = tf.nn.max_pool(x, ksize=[2,2], strides=[2,2], padding='SAME')
        # print(x.shape)

        x = tf.nn.conv2d(x, self.C3, strides=[1,1], padding='SAME')
        mean, variance = tf.nn.moments(x, [0,1,2])
        x = tf.nn.batch_normalization(x, mean=mean, variance=variance, variance_epsilon=1e-5, offset=None, scale=None)
        x = tf.nn.relu(x)
        # print(x.shape)

        x = tf.reshape(x, [self.batch_size, -1])
        # print(x.shape)


        #dense layer 1 
        x = x @ self.W1 + self.b1
        x = tf.nn.dropout(x, rate=0.3)
        # #dense layer 2
        x = x @ self.W2 + self.b2
        x = tf.nn.dropout(x, rate=0.3)
        #dense layer 3
        x = x @ self.W3 + self.b3

        x = tf.nn.softmax(x)
        
        return x

    def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        :param logits: during training, a matrix of shape (batch_size, self.num_classes) 
        containing the result of multiple convolution and feed forward layers
        Softmax is applied in this function.
        :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
        :return: the loss of the model as a Tensor
        """
        x = tf.nn.softmax_cross_entropy_with_logits(labels, logits)
        x = tf.math.reduce_mean(x)

        return x

    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels – no need to modify this.
        :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)

        NOTE: DO NOT EDIT
        
        :return: the accuracy of the model as a Tensor
        """
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs 
    and labels - ensure that they are shuffled in the same order using tf.gather.
    To increase accuracy, you may want to use tf.image.random_flip_left_right on your
    inputs before doing the forward pass. You should batch your inputs.
    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training), 
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training), 
    shape (num_labels, num_classes)
    :return: Optionally list of losses per batch to use for visualize_loss
    '''
    num_ex, _, _, _ = train_inputs.shape
    num_batches = num_ex // model.batch_size

    for i in range(num_batches):
        batch_inputs, batch_labels = get_batch(train_inputs, train_labels, model.batch_size, i * model.batch_size)
        
        with tf.GradientTape() as tape:
            predictions = model.call(batch_inputs)
            loss = model.loss(predictions, batch_labels)

        model.loss_list.append(loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    print(model.accuracy(predictions, batch_labels))
    return model.loss_list

def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. You should NOT randomly 
    flip images or do any extra preprocessing.
    :param test_inputs: test data (all images to be tested), 
    shape (num_inputs, width, height, num_channels)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :return: test accuracy - this should be the average accuracy across
    all batches
    """
    num_ex, _, _, _ = test_inputs.shape
    num_batches = num_ex // model.batch_size

    accur = 0
    for i in range(num_batches):
        batch_inputs, batch_labels = get_batch(test_inputs, test_labels, model.batch_size, i * model.batch_size)
    
        predictions = model.call(batch_inputs)
        accur += model.accuracy(predictions, batch_labels)

    return accur/num_batches


def visualize_loss(losses): 
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list 
    field 

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up 
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()  


def visualize_results(image_inputs, probabilities, image_labels, first_label, second_label):
    """
    Uses Matplotlib to visualize the correct and incorrect results of our model.
    :param image_inputs: image data from get_data(), limited to 50 images, shape (50, 32, 32, 3)
    :param probabilities: the output of model.call(), shape (50, num_classes)
    :param image_labels: the labels from get_data(), shape (50, num_classes)
    :param first_label: the name of the first class, "cat"
    :param second_label: the name of the second class, "dog"

    NOTE: DO NOT EDIT

    :return: doesn't return anything, two plots should pop-up, one for correct results,
    one for incorrect results
    """
    # Helper function to plot images into 10 columns
    def plotter(image_indices, label): 
        nc = 10
        nr = math.ceil(len(image_indices) / 10)
        fig = plt.figure()
        fig.suptitle("{} Examples\nPL = Predicted Label\nAL = Actual Label".format(label))
        for i in range(len(image_indices)):
            ind = image_indices[i]
            ax = fig.add_subplot(nr, nc, i+1)
            ax.imshow(image_inputs[ind], cmap="Greys")
            pl = first_label if predicted_labels[ind] == 0.0 else second_label
            al = first_label if np.argmax(
                image_labels[ind], axis=0) == 0 else second_label
            ax.set(title="PL: {}\nAL: {}".format(pl, al))
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)
        
    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = image_inputs.shape[0]

    # Separate correct and incorrect images
    correct = []
    incorrect = []
    for i in range(num_images): 
        if predicted_labels[i] == np.argmax(image_labels[i], axis=0): 
            correct.append(i)
        else: 
            incorrect.append(i)

    plotter(correct, 'Correct')
    plotter(incorrect, 'Incorrect')
    plt.show()


def main():
    '''
    Read in CIFAR10 data (limited to 2 classes), initialize your model, and train and 
    test your model for a number of epochs. We recommend that you train for
    10 epochs and at most 25 epochs. 
    
    CS1470 students should receive a final accuracy 
    on the testing examples for cat and dog of >=70%.
    
    CS2470 students should receive a final accuracy 
    on the testing examples for cat and dog of >=75%.
    
    :return: None
    '''
    train_inputs, train_labels = get_data("../data/train", 5, 6)
    test_inputs, test_labels = get_data("../data/test", 5, 6)

    # print(train_inputs.shape, train_labels.shape)
    # print(test_inputs.shape, test_labels.shape)

    model = Model()
    # print(train_input.shape, train_labels.shape)
    for i in range(10):
        rands = tf.random.shuffle(np.arange(10000))
        train_inputs = tf.gather(train_inputs, rands)
        train_labels = tf.gather(train_labels, rands)
        
        train(model, train_inputs, train_labels)
    
    visualize_loss(model.loss_list)

    test_rands = tf.random.shuffle(np.arange(2000))
    test_inputs = tf.gather(test_inputs, test_rands)
    test_labels = tf.gather(test_labels, test_rands)
    print(test(model, test_inputs, test_labels))
    return


if __name__ == '__main__':
    main()
