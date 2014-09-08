# -*- coding: utf-8 -*-
"""
Created on Tue May 14 20:32:38 2013

@author: alberto
"""

import numpy as np
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn import ensemble
from sklearn.svm import SVR
from matplotlib import pyplot as plt
from sklearn.datasets import  load_boston
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from matplotlib.font_manager import FontProperties

# data = load_diabetes()
data = load_boston()
X = data.data
y = data.target

boston = datasets.load_boston()
X, y = shuffle(boston.data, boston.target, random_state=13)
X = X.astype(np.float32)
offset = int(X.shape[0] * 0.9)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

#for j in range(0,13):
#    plt.subplot(4,4,j+1)
#
#    for i in range(0,506):
#        
#        plt.scatter(y[i], X[i, j], c='red')
#    
#plt.show()

#for j in range(0,1):
    #plt.subplot(4,4,j+1)

#for i in range(0,506):
#        plt.scatter(X[i,5], X[i,12], c='red')
#    
#plt.show()



nCases = len(y)
nTrain = np.floor(nCases * 0.7)
import random
ids = range(nCases)
random.shuffle(ids)

#for i in range(len(testY))
#    plt.scatter(trainY, )

trainX,trainY,testX,testY = [],[],[],[]
for i, idx in enumerate(ids):
    if i < nTrain:
        trainX.append(X[idx])
        trainY.append(y[idx])
    else:
        testX.append(X[idx])
        testY.append(y[idx])


v = [ [0 for i in range(0,1) ] for j in range(len(trainX)) ]

for i in range(0,1):
    for j in range(len(trainX)):
        v[i].append(trainX[j][i])


#dTrain= [ [ 0 for i in range(di+1) ] for j in range(len(X_train)) ]


params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 1,
          'learning_rate': 0.01, 'loss': 'ls'}
clfa = ensemble.GradientBoostingRegressor(**params)

clfa.fit(trainX, trainY)
preGradient = clfa.predict(testX)
mseGradient = metrics.mean_absolute_error(testY, preGradient)
mseSGradient = metrics.mean_squared_error(testY, preGradient)



feature_importance = clfa.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())
print feature_importance
feature_importance[0] = feature_importance[0] * 2
print feature_importance
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
#
#svr = SVR(kernel='linear', C=1.0, epsilon=0.2)
#log = LinearRegression()
#regressor = DecisionTreeRegressor(random_state=0)
#
#clf = Ridge(alpha=1.)
#clf.fit(trainX, trainY)
#pred = np.round(clf.predict(testX))
#sRid = metrics.mean_absolute_error(testY, pred)
#
#svr.fit(trainX, trainY)
#log.fit(trainX, trainY)
#regressor.fit(trainX, trainY)
#
#predLog = log.predict(testX)
#predSvr = svr.predict(testX)
#predregressor = regressor.predict(testX)
#
#sL = metrics.mean_absolute_error(testY, predLog)
#sS= metrics.mean_absolute_error(testY, predSvr)
#sR = metrics.mean_absolute_error(testY, predregressor)
#
#sSL = metrics.mean_squared_error(testY, predLog)
#sSS = metrics.mean_squared_error(testY, predSvr)
#sSR = metrics.mean_squared_error(testY, predregressor)
#sSRid = metrics.mean_squared_error(testY, pred)
#
#
#print "  "
#print "Error medio absoluto:  "
#print "   Linear        SVR        DecisionTree      Ridge      Gradient"
#print sL, sS, sR, sRid, mseGradient
#
#print "   "
#print "Error medio cuadratico:   "
#print "   Linear        SVR        DecisionTree      Ridge      Gradient"
#print sSL, sSS, sSR, sSRid, mseSGradient
#
#
#
#plt.subplot(2,3,1)
#plt.hist(data.target)
#plt.xlabel('price ($1000s)')
#plt.ylabel('count')
#
#plt.subplot(2,3,2)
#plt.plot(testY, testY, label='true data')
#plt.plot(testY, predSvr, 'co', label='SVR')
#plt.legend(bbox_to_anchor=(1.05, 1), borderaxespad=0., prop = FontProperties(size='smaller'))
#
#plt.subplot(2,3,3)
#plt.plot(testY, testY, label='true data')
#plt.plot(testY, predLog, 'mo', label='Reg Lineal')
#plt.legend(bbox_to_anchor=(1.05, 1), borderaxespad=0., prop = FontProperties(size='smaller'))
#
#plt.subplot(2,3,4)
#plt.plot(testY, testY, label='true data')
#plt.plot(testY, pred, 'bo', label='Ridge')
#plt.legend(bbox_to_anchor=(1.05, 1), borderaxespad=0., prop = FontProperties(size='smaller'))
#
#plt.subplot(2,3,5)
#plt.plot(testY, testY, label='true data')
#plt.plot(testY, predregressor, 'yo', label='Decision Tree')
#
#plt.legend(bbox_to_anchor=(1.05, 1), borderaxespad=0., prop = FontProperties(size='smaller'))
#
#plt.subplot(2,3,6)
#plt.plot(testY, testY, label='true data')
#plt.plot(testY, preGradient, 'ro', label='Gradient')
#
#plt.legend(bbox_to_anchor=(1.05, 1), borderaxespad=0., prop = FontProperties(size='smaller'))
#plt.show()
#
#plt.barh(pos, feature_importance[sorted_idx], align='center')
#plt.yticks(pos, boston.feature_names[sorted_idx])
#plt.xlabel('Relative Importance')
#plt.title('Variable Importance')
#plt.show()