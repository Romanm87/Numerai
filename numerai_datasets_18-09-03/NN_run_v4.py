# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 06:49:14 2018

@author: Roman
"""
### 9/3/2018
### model trained again on bad eras (for loop through NNs)

# Import packages (NN from Coursera C2/Assign4)
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
import time
from sklearn import metrics, preprocessing, linear_model
import sys
import warnings
#from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict
import os
os.chdir('c:\\Users\\Roman\\Documents\\Projects\\Numerai\\numerai_datasets_18-09-03')

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
x_tournament = np.array(tournament[features].T)
ids = tournament['id']

### tune hyperparameters:
learning_rate = 1e-5
num_epochs = 300
minibatch_size = 500
keep_prob = 0.8

### train and evaluate NN:
import NN_functions_v2 as nn

start = time.time()
parameters, costs, val_costs = nn.model(X_trainNN, Y_trainNN, x_validation, y_validation, keep_prob, learning_rate, num_epochs, minibatch_size)
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
print("validation loss: " + str(logloss))

# add predictions to validation pd.df:
if not sys.warnoptions:
    warnings.simplefilter("ignore")
validation['pred'] = pd.Series(y_hat_val[0,:])
eras = validation.era.unique()
dfs = {}
for era in eras[:-1]:
    dfs[era] = validation[validation['era'] == era]
logloss_era = pd.Series()
for era in eras[:-1]:
    logloss_era[era] = metrics.log_loss(dfs[era]['target_bernie'], dfs[era]['pred'])
consistency = round(100 * sum(logloss_era < -np.log(0.5)) / logloss_era.shape[0], 2)
print("Consistency: " + str(consistency))
logloss_era = logloss_era.sort_values(ascending=False)
print("eras with too high logloss: ")
for i in range(sum(logloss_era >= -np.log(0.5))):
    print(logloss_era.index[logloss_era > -np.log(0.5)][i] +": "+ str(round(logloss_era[logloss_era > -np.log(0.5)][i],4)))

plt.plot(costs[3:300], color = 'blue')
plt.plot(val_costs[3:300], color = 'red')
plt.title("costs")
plt.grid()
plt.show()

### train additional model for bad eras:
# identify bad eras:
train['pred'] = pd.Series(y_hat_train[0,:])
eras_train = train['era'].unique()
dfs_train = {}
for era in eras_train:
    dfs_train[era] = train[train['era'] == era]
era_train_df = pd.DataFrame(index = eras_train, columns = ['logloss', 'cluster'])
for era in eras_train:
    era_train_df.loc[era, 'logloss'] = metrics.log_loss(dfs_train[era]['target_bernie'], dfs_train[era]['pred'])
era_train_df.loc[:, 'cluster'] = clusters #<- determine clusters with Cluster.py
high_logloss = sum(era_train_df['logloss'] >= -np.log(0.5))
consistency_train = round(100 * (1 - high_logloss / era_train_df.shape[0]), 2)
print("Consistency: " + str(consistency_train))
era_train_df = era_train_df.sort_values('logloss', ascending=False) # <-sort with highes logloss first
print("eras with too high logloss: ")
print(era_train_df.iloc[1:high_logloss, :])

# boxplot for logloss by cluster
cluster_cat = sorted(era_train_df.loc[:, 'cluster'].unique())
vectors = []
for cluster in cluster_cat:
    vectors.append(list(era_train_df.loc[era_train_df['cluster'] == cluster, 'logloss']))
plt.boxplot(vectors, labels=list(range(7)))
plt.grid()
plt.title("Logloss by Cluster")
plt.show()

# prepare data (cluster with bad eras) 1, 6:
parameters_era = []
y_hat_train_all_era = []
y_hat_val_era = []
for cl in [1, 6]:   #<- enter all bad clusters in vector
    eras_cl = era_train_df[era_train_df['cluster'] == cl].index
    train_era = train[train['era'].isin(list(eras_cl))] 
    train_bernie_era = train_era.drop(['era','pred'], axis=1)
    X_trainNN_era = np.array(train_bernie_era[features].T)
    Y_trainNN_era = np.array(train_bernie_era['target_bernie']).reshape(1, -1)

### tune hyperparameters (bad eras):
    learning_rate = 5e-5
    num_epochs = 15
    minibatch_size = 500
    keep_prob = 0.8

### train NN (bad eras):
    import NN_functions as nn

    start = time.time()
    parameters_, _ = nn.model(X_trainNN_era, Y_trainNN_era, keep_prob, learning_rate, num_epochs, minibatch_size)
    parameters_era.append(parameters_)
    end = time.time()
    #print("number of examples: " + str(X_trainNN_era.shape[1]))
    #print("number of epochs: " + str(num_epochs))
    #print("minibatch size: " + str(minibatch_size))
    #print("number of minibatches: " + str(round(X_trainNN_era.shape[1]/minibatch_size)))
    #print("number of iterations: " + str(num_epochs * round(X_trainNN_era.shape[1]/minibatch_size)))
    #print("dropout probability: " + str(keep_prob))
    #print("training time: " + str(round(end - start)) + " seconds")

    y_hat_train_era = nn.sigmoid(nn.pred(X_trainNN_era, parameters_))
    #accuracy_train_eraI = np.sum([np.round(y_hat_train_eraI) == Y_trainNN_era]) / Y_trainNN_era.shape[1]
    logloss_train_era = metrics.log_loss(pd.Series(Y_trainNN_era[0,:]), y_hat_train_era[0,:])
    print("training loss cluster " + str(cl) + ": " +str(logloss_train_era))

    #plt.plot(costs)
    #plt.title("costs")
    #plt.show()

    # calculate predictions on bad era model:
    y_hat_train_all_era.append( nn.sigmoid(nn.pred(X_trainNN, parameters_)) )
    #logloss_train_all_eraI = metrics.log_loss(pd.Series(Y_trainNN[0,:]), y_hat_train_all_eraI[0,:])
    #print("training loss: " + str(logloss_train_all_eraI))
    y_hat_val_era.append( nn.sigmoid(nn.pred(x_validation, parameters_)) )

# combine models
y_hat_train_combined = 0.775*y_hat_train + 0.075*y_hat_train_all_era[0] + 0.15*y_hat_train_all_era[1]
y_hat_val_combined = 0.775*y_hat_val + 0.075*y_hat_val_era[0] + 0.15*y_hat_val_era[1]

#evaluate combined model:
logloss_train_combined = metrics.log_loss(pd.Series(Y_trainNN[0,:]), y_hat_train_combined[0,:])
print("training loss: " + str(logloss_train_combined))
logloss = metrics.log_loss(pd.Series(y_validation[0,:]), y_hat_val_combined[0,:])
print("validation loss: " + str(logloss))

# identify bad eras after combination:
train['pred2'] = pd.Series(y_hat_train_combined[0,:])
dfs_train = {}
for era in eras_train:
    dfs_train[era] = train[train['era'] == era]
era_train_df2 = pd.DataFrame(index = eras_train, columns = ['logloss', 'cluster'])
for era in eras_train:
    era_train_df2.loc[era, 'logloss'] = metrics.log_loss(dfs_train[era]['target_bernie'], dfs_train[era]['pred2'])
era_train_df2.loc[:, 'cluster'] = clusters #<- determine clusters with Clusters.py
high_logloss = sum(era_train_df2['logloss'] >= -np.log(0.5))
consistency_train = round(100 * (1 - high_logloss / era_train_df2.shape[0]), 2)
print("Consistency: " + str(consistency_train))
era_train_df2 = era_train_df2.sort_values('logloss', ascending=False) # <-sort with highes logloss first
print("eras with too high logloss: ")
print(era_train_df2.iloc[1:high_logloss, :])

# boxplot with new model:
cluster_cat = sorted(era_train_df2.loc[:, 'cluster'].unique())
vectors = []
for cluster in cluster_cat:
    vectors.append(list(era_train_df2.loc[era_train_df2['cluster'] == cluster, 'logloss']))
plt.boxplot(vectors, labels=list(range(7)))
plt.grid()
plt.title("Logloss by Cluster (after combination)")
plt.show()

# calculate tournament predictions:
y_hat_tourn_reg = nn.sigmoid(nn.pred(x_tournament, parameters))
y_hat_tourn_eraI = nn.sigmoid(nn.pred(x_tournament, parameters_era[0]))
y_hat_tourn_eraII = nn.sigmoid(nn.pred(x_tournament, parameters_era[1]))
y_hat_tourn_combined = 0.775*y_hat_tourn_reg + 0.075*y_hat_tourn_eraI + 0.15*y_hat_tourn_eraII

# Write to csv:
#y_hat_tournament = nn.sigmoid(nn.pred(x_tournament, parameters))
y_hat_tournament_df = pd.DataFrame(y_hat_tourn_combined)
joined = pd.DataFrame(ids).join(np.transpose(y_hat_tournament_df))
joined.columns = ['id', 'probability_bernie'] #<- check
joined.to_csv("bernie_submission_RM.csv", index=False)





