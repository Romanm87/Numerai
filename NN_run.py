# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 06:49:14 2018

@author: Roman
"""
# Import packages (NN from Coursera C2/Assign4)
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
import time
from sklearn import metrics, preprocessing, linear_model
#from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict
import os
os.chdir('c:\\Users\\Roman\\Documents\\Projects\\Numerai\\numerai_datasets')

### load and transform data
train = pd.read_csv('numerai_training_data.csv', header=0)
# Tournament data is the data that Numerai uses to evaluate your model.
tournament = pd.read_csv('numerai_tournament_data.csv', header=0)
# Tournament contains validation data, test data, live data. -> use Validation to test model locally
# Validation is used to test your model locally so we separate that.
validation = tournament[tournament['data_type']=='validation']
train_bernie = train.drop([
    'id', 'era', 'data_type',
    'target_charles', 'target_elizabeth',
    'target_jordan', 'target_ken'], axis=1)
features = [f for f in list(train_bernie) if "feature" in f]
X_trainNN = np.array(train_bernie[features].T)   #rows: number of features, cols: number of examples
Y_trainNN = np.array(train_bernie['target_bernie']).reshape(1, -1)
x_validation = np.array(validation[features].T)
y_validation = np.array(validation['target_bernie']).reshape(1, -1)
ids = tournament['id']

### tune hyperparameters:
learning_rate = 1e-5 
num_epochs = 500
minibatch_size = 200
keep_prob = 0.8

### train and evaluate NN:
import NN_functions as nn

start = time.time()
parameters, costs = nn.model(X_trainNN, Y_trainNN, keep_prob, learning_rate, num_epochs, minibatch_size)
end = time.time()

print("number of examples: " + str(X_trainNN.shape[1]))
print("number of epochs: " + str(num_epochs))
print("minibatch size: " + str(minibatch_size))
print("number of minibatches: " + str(round(X_trainNN.shape[1]/minibatch_size)))
print("number of iterations: " + str(num_epochs * round(X_trainNN.shape[1]/minibatch_size)))
print("dropout probability: " + str(keep_prob))
print("training time: " + str(round(end - start)) + " seconds")

y_hat_train = nn.sigmoid(nn.pred(X_trainNN, parameters))
accuracy_train = np.sum([np.round(y_hat_train) == Y_trainNN]) / Y_trainNN.shape[1]
print("train accuracy: ", accuracy_train)

y_hat_val = nn.sigmoid(nn.pred(x_validation, parameters))
accuracy_val = np.sum([np.round(y_hat_val) == y_validation]) / y_validation.shape[1]
print("validation accuracy: ", accuracy_val)

print("training loss: " + str(costs[-1]))
logloss = metrics.log_loss(pd.Series(y_validation[0,:]), y_hat_val[0,:])
print("validation loss: ", logloss)

plt.plot(costs)
plt.title("costs")
plt.show()


    










