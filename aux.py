# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 18:41:06 2013

@author: alberto
"""

def pruebaModelos():
    
    gra = ensemble.GradientBoostingRegressor(n_estimators=500, max_depth=4, min_samples_split=1, learning_rate=0.01, loss='ls')
    svr = SVR(kernel='linear', C=1.0, epsilon=0.2)
    reg = LinearRegression()
    dtr = DecisionTreeRegressor()
    
    cla=(gra, svr, reg, dtr)
    
    print "Validazion cruzada para los 5 modelos(en orden: GradientBoosting, SVR, LinearRegression, TreeRegressor)"
    for c in cla:
        #VALIDACION CRUZADA
        mae=mse=r2=0
        kf = KFold(len(boston_Y), n_folds=10, indices=True)
        for train, test in kf:
            trainX, testX, trainY, testY=boston_X[train], boston_X[test], boston_Y[train], boston_Y[test]
            
            c.fit(trainX, trainY)
            prediccion=c.predict(testX)
            
            mae+=metrics.mean_absolute_error(testY, prediccion)
            mse+=metrics.mean_squared_error(testY, prediccion)
            r2+=metrics.r2_score(testY, prediccion)
        print 'Error abs: ', mae/len(kf), 'Error cuadratico: ', mse/len(kf), 'R cuadrado: ', r2/len(kf)
        #FIN FOR VALIDACION CRUZADA
    #FIN FOR CLASIFICADORES
#FIN pruebaModelos




def tecnicaSVR():
    
    parametros = [{'kernel':'linear', 'C':1.0, 'epsilon':0.2},
                  {'kernel':'rbf', 'degree':3, 'gamma':.0001, 'C':1.0, 'epsilon':0.2},
                  {'kernel':'sigmoid', 'degree':4, 'gamma':0.1, 'C':1.0, 'epsilon':0.1}]

    mae=mse=r2=0
    
    for c in parametros:
        clf = SVR(**c)
        #VALIDACION CRUZADA
        mae=mse=r2=0
        kf = KFold(len(boston_Y), n_folds=10, indices=True)
        for train, test in kf:
            trainX, testX, trainY, testY=boston_X[train], boston_X[test], boston_Y[train], boston_Y[test]
            
            clf.fit(trainX, trainY)
            prediccion=clf.predict(testX)
            
            mae+=metrics.mean_absolute_error(testY, prediccion)
            mse+=metrics.mean_squared_error(testY, prediccion)
            r2+=metrics.r2_score(testY, prediccion)
            
        print 'Error abs: ', mae/len(kf), 'Error cuadratico: ', mse/len(kf), 'R cuadrado: ', r2/len(kf)
        mae=mse=r2=0



def ajustesParametros():
    parametros_gradient = [{'n_estimators': 100, 'max_depth': 2, 'min_samples_split': 1, 'learning_rate': 0.01, 'loss': 'ls'},
                           {'n_estimators': 200, 'max_depth': 3, 'min_samples_split': 2, 'learning_rate': 0.1, 'loss': 'huber'},
                           {'n_estimators': 300, 'max_depth': 2, 'min_samples_split': 1, 'learning_rate': 0.01, 'loss': 'ls'},
                           {'n_estimators': 400, 'max_depth': 3, 'min_samples_split': 2, 'learning_rate': 0.1, 'loss': 'huber'}]
#    parametros_svr = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]},
#                      {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
#    parametros_tree = [{'max_depth':[1]},
#                       {'max_depth':[10]}]
    #parametros_linear = []

    for c in parametros_gradient:
        print c
        clf = ensemble.GradientBoostingRegressor(**c)
        #VALIDACION CRUZADA
        mae=mse=r2=0
        kf = KFold(len(boston_Y), n_folds=10, indices=True)
        for train, test in kf:
            trainX, testX, trainY, testY=boston_X[train], boston_X[test], boston_Y[train], boston_Y[test]
            
            clf.fit(trainX, trainY)
            prediccion=clf.predict(testX)
            
            mae+=metrics.mean_absolute_error(testY, prediccion)
            mse+=metrics.mean_squared_error(testY, prediccion)
            r2+=metrics.r2_score(testY, prediccion)
        print 'Error abs: ', mae/len(kf), 'Error cuadratico: ', mse/len(kf), 'R cuadrado: ', r2/len(kf)






