## Miniflow

Naive recreation of the core functionality of TensorFlow

### Project structure
* miniflow.py: Implementation of:
    * The different nodes (Input, Linear Transform, Sigmoid Activation Function and MSE )of the Computational Graph
    * Thin wrapper to ensure the connectivity of the nodes and edges
    * Forward and backward propgation mechanisms, and,
    * Stochastic Gradient Descent
* nn.py: Creates a 2 layer NN and trains a Linear Regression Model over the Boston Housing dataset using backprop and stochastic gradient descent to predict housing prices.