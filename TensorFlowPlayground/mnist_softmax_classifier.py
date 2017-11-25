__author__ = 'sidroopdaska'
""" Softmax Classifier to classify the MNIST data set"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from TensorFlowPlayground.helpers import ModelTrainingLogger

# Get the MNIST data set
mnist = input_data.read_data_sets("mnist_dataset/", one_hot=True)

test_images = mnist.test.images
test_labels = mnist.test.labels.astype(np.float32)

val_images = mnist.validation.images
val_labels = mnist.validation.labels.astype(np.float32)

# Specify the Hyper parameter values
learning_rate = tf.placeholder(tf.float32)
n_features = 784
n_labels = 10
batch_size = 128
epochs = 200

# Define the Softmax classifier/ 1 layer NN architecture
features = tf.placeholder(tf.float32, [None, n_features])
weights = tf.Variable(tf.random_normal([n_features, n_labels]))
biases = tf.Variable(tf.zeros([n_labels]))
labels = tf.placeholder(tf.float32, [None, n_labels])

logits = tf.matmul(features, weights) + biases

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
optimiser = tf.train.AdamOptimizer(learning_rate)\
                .minimize(cost)

correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

batches = int(mnist.train.num_examples / batch_size)

logger = ModelTrainingLogger(epochs)
# Run the training session and plot the learning curves
with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    for epoch_i in range(epochs):
        for batch_i in range(batches):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            session.run(optimiser,
                        feed_dict={
                            features: batch_x,
                            labels: batch_y,
                            learning_rate: 0.005,
                        })
        curr_cost = session.run(cost, feed_dict={features: batch_x, labels: batch_y})

        tr_accuracy = session.run(
            accuracy,
            feed_dict={features: mnist.train.images, labels: mnist.train.labels})
        val_accuracy = session.run(
            accuracy,
            feed_dict={features: val_images, labels: val_labels})
        logger.update(epoch_i, curr_cost, tr_accuracy*100, val_accuracy*100)

    test_accuracy = session.run(
            accuracy,
            feed_dict={features: test_images, labels: test_labels})
    print("Test Accuracy: {}".format(test_accuracy))
