__author__ = 'sidroopdaska'
""" 2 layer NN to classify the MNIST data set """

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from TensorFlowPlayground.helpers import ModelTrainingLogger

# Get the data set
mnist = input_data.read_data_sets("mnist_dataset/", one_hot=True)

# Data set pre processing

# Init Hyper params
learn_rate = tf.placeholder(tf.float32)
epochs = 20
batch_size = 128

n_features = 784
n_labels = 10
n_hidden_units = 256
keep_probab = tf.placeholder(tf.float32)

# Define the arch of the 2 layer NN
features = tf.placeholder(tf.float32, [None, n_features])
labels = tf.placeholder(tf.float32, [None, n_labels])

weights = {
    'hidden': tf.Variable(tf.truncated_normal([n_features, n_hidden_units])),
    'out': tf.Variable(tf.truncated_normal([n_hidden_units, n_labels]))
}

biases = {
    'hidden': tf.Variable(tf.zeros(n_hidden_units)),
    'out': tf.Variable(tf.zeros(n_labels))
}

hidden_layer = tf.matmul(features, weights['hidden']) + biases['hidden']
hidden_layer = tf.nn.relu(hidden_layer)
hidden_layer = tf.nn.dropout(hidden_layer, keep_prob=keep_probab)

logits = tf.matmul(hidden_layer, weights['out']) + biases['out']

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

optimiser = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss)
correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

n_batches = int(np.ceil(mnist.train.num_examples/batch_size))
logger = ModelTrainingLogger(epochs)

# Run the session
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    batch_features, batch_labels = None, None

    for epoch_i in range(epochs):
        for batch_i in range(n_batches):
            batch_features, batch_labels = mnist.train.next_batch(batch_size)

            _ = session.run(
                optimiser,
                feed_dict={
                    features: batch_features,
                    labels: batch_labels,
                    learn_rate: 0.5,
                    keep_probab: 0.5
                }
            )

        l = session.run(loss, feed_dict={features: batch_features, labels: batch_labels, keep_probab: 0.5})
        tr_acc = session.run(accuracy, feed_dict={features: mnist.train.images,
                                                  labels: mnist.train.labels, keep_probab:1.0})
        val_acc = session.run(accuracy, feed_dict={features: mnist.validation.images,
                                                   labels: mnist.validation.labels, keep_probab: 1.0})
        logger.update(epoch_i, l, tr_acc*100, val_acc*100)

    print("Test Accuracy: {0}".format(session.run(accuracy, feed_dict={
        features: mnist.test.images,
        labels: mnist.test.labels,
        keep_probab: 1.0
    })))


