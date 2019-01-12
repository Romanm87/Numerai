# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 07:11:02 2018

@author: Roman
"""
import pandas as pd
import numpy as np
from matplotlib import *
import matplotlib.pyplot as plt 
from sklearn import linear_model, cluster, preprocessing, decomposition
from scipy.spatial.distance import cdist
import os
os.chdir('c:\\Users\\Roman\\Documents\\Projects\\Numerai\\numerai_datasets_18-09-03')

### Method 2: logistic regression per era

train_ = pd.read_csv('numerai_training_data.csv', header=0)
train_ = train_.drop(['id', 'data_type', 'target_charles', 'target_elizabeth','target_jordan', 'target_ken'], axis=1)
features = [f for f in list(train_) if "feature" in f]
eras = train_.era.unique()

# dict that groups training data per era:
train_era = {}
for era in eras:
    train_era[era] = train_[train_['era'] == era]
    train_era[era] = train_era[era].drop(['era'], axis=1)

model = linear_model.LogisticRegression()
 
# store regression coefficitent for each era:
lr_era = np.empty((len(train_era), len(features)))
for i in range(lr_era.shape[0]):
    X = train_era[eras[i]][features]
    Y = train_era[eras[i]]['target_bernie']
    model.fit(X, Y)
    lr_era[i, :] = model.coef_

### scree plot to find number of components
X_std = preprocessing.StandardScaler(copy=True).fit_transform(lr_era) #equal var and mean=0 for coeff of each feature
pca = decomposition.PCA(n_components=15).fit(X_std) # check not all 50 components to make plot more readable

# explained variance:
plt.plot(pca.explained_variance_ratio_)
plt.title("Scree Plot")
plt.xlabel('number of components')
plt.ylabel('explained variance')
plt.grid()
plt.show()

# cumlative explained variance:
plt.plot(np.cumsum(pca.explained_variance_ratio_)) # <- use for cumulative explained variance
plt.title("Scree Plot Cumulative")
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.grid()
plt.show()
# -> 7 components explain >65% of variance (sufficient?)

pca = decomposition.PCA(n_components = 7).fit_transform(lr_era)
pca_arr = np.array(pca)

dist = []
K = range(1, 14)
for k in K:
    model = cluster.KMeans(n_clusters = k).fit(pca_arr)
    dist.append(sum(np.min(cdist(pca_arr, model.cluster_centers_, 'euclidean'), axis=1)) / pca_arr.shape[0])
        
plt.plot(K, dist, 'bx-')
plt.title("Search for K - Ellbow-method")
plt.xlabel('k')
plt.ylabel('distortion')
plt.grid()
plt.show()
# -> choose 7 clusters

model = cluster.KMeans(n_clusters = 7).fit(pca_arr)
clusters = model.predict(pca)


