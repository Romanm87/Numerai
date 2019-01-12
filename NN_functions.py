# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 06:49:14 2018

@author: Roman
"""
from __main__ import *

##### sigmoid function:
def sigmoid(z):
    x = tf.placeholder(tf.float32, name = 'x')
    sigmoid = tf.sigmoid(x)
    with tf.Session() as sess:
        result = sess.run(sigmoid, feed_dict = {x: z})
    return result

### loss function: (logits: computed output, labels: labled output)
def cost(logits, labels):
    z = tf.placeholder(tf.float32, name = 'z')
    y = tf.placeholder(tf.float32, name = 'y')
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits = z, labels = y)
    sess = tf.Session()
    cost = sess.run(cost, feed_dict = {z: logits, y: labels})
    sess.close()
    return cost

### create placeholders for input and output:
def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, shape = (n_x, None), name = 'Placeholder')
    Y = tf.placeholder(tf.float32, shape = (n_y, None), name = 'Placeholder')
    return X, Y

# initialize parameters
def initialize_parameters():
    W1 = tf.get_variable("W1", [35, len(features)], initializer = tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", [35, 1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [17, 35], initializer = tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", [17, 1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [1, 17], initializer = tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3", [1, 1], initializer = tf.zeros_initializer())
    #W4 = tf.get_variable("W4", [1, 12], initializer = tf.contrib.layers.xavier_initializer())
    #b4 = tf.get_variable("b4", [1, 1], initializer = tf.zeros_initializer())   
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}#, "W4": W4, "b4": b4}
    return parameters

# forward propagation:
def forward_propagation(X, parameters, keep_prob):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    #W4 = parameters['W4']
    #b4 = parameters['b4']
    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    #keep_prob = tf.placeholder("float")
    A1_dropout = tf.nn.dropout(A1, keep_prob)
    Z2 = tf.add(tf.matmul(W2, A1_dropout), b2)
    A2 = tf.nn.relu(Z2)
    A2_dropout = tf.nn.dropout(A2, keep_prob)
    Z3 = tf.add(tf.matmul(W3, A2_dropout), b3)
    #A3 = tf.nn.relu(Z3)
    #Z4 = tf.add(tf.matmul(W4, A3), b4)
    return Z3

def compute_cost(Z, Y):
    logits = tf.transpose(Z)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels))
    return cost

# minibatches:
def create_minibatches(X, Y, minibatch_size):
    n_batches = round(X.shape[1] / minibatch_size)
    minibatch_x = []; minibatch_y = []
    for i in range(n_batches-1):
        batch_x = X[:, i*minibatch_size : ((i+1)*minibatch_size)]
        batch_y = Y[:, i*minibatch_size : ((i+1)*minibatch_size)]
        minibatch_x.append(batch_x); minibatch_y.append(batch_y)
    last_batch_x = X[:, ((n_batches-1)*minibatch_size):X.shape[1]]
    last_batch_y = Y[:, ((n_batches-1)*minibatch_size):Y.shape[1]]
    minibatch_x.append(last_batch_x); minibatch_y.append(last_batch_y)
    return minibatch_x, minibatch_y
    
def model(X_train, Y_train, keep_prob, learning_rate, num_epochs, minibatch_size, print_cost = True):
    ops.reset_default_graph() 
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []
    X, Y = create_placeholders(n_x, n_y)
    parameters = initialize_parameters()
    Z = forward_propagation(X, parameters, keep_prob)
    cost = compute_cost(Z, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            epoch_cost = 0.
            minibatch_x, minibatch_y = create_minibatches(X_train, Y_train, minibatch_size)
            for i in range(len(minibatch_x)):
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X:minibatch_x[i], Y:minibatch_y[i]}) 
                epoch_cost += minibatch_cost / len(minibatch_x)
                if i % 50 == 0:
                    print(str(round(1/num_epochs * (epoch+(i/len(minibatch_x))) * 100, 2)) + "%")
            costs.append(epoch_cost)
            if print_cost == True and epoch % 1 == 0:
                print ("cost after epoch %i: %f" % (epoch, epoch_cost))
        parameters = sess.run(parameters)
        return parameters, costs

def pred(x, parameters):
    sess = tf.Session()
    X, Y = create_placeholders(len(features), 1)
    z = forward_propagation(X, parameters, keep_prob=1)
    result = sess.run(z, feed_dict = {X: x})
    sess.close()
    return result








