# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 12:38:15 2013

@author: alberto
"""

from sklearn import datasets, ensemble, metrics, preprocessing
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn import tree, utils
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from matplotlib.font_manager import FontProperties
import numpy as np
import pylab as pl

def normalizacion():
    scaler = preprocessing.StandardScaler().fit(boston_X)
    scaler.mean_
    scaler.std_
    scaler.transform(boston_X) 
#FIN FUNCION normalizacion

def aprendizajePorCaractGradient(caract):

    clf = ensemble.GradientBoostingRegressor(n_estimators=350, max_depth=2, learning_rate=0.1, loss='ls', subsample=0.5)

    #VALIDACION CRUZADA
    mae=mse=r2=0
    kf = KFold(len(boston_Y), n_folds=10, indices=True)
    for train, test in kf:
        trainX, testX, trainY, testY=boston_X[train], boston_X[test], boston_Y[train], boston_Y[test]
        
        nCar=len(caract)
        train=np.zeros((len(trainX), nCar))
        test=np.zeros((len(testX), nCar))
        trainYNuevo=trainY
        
        for i in range(nCar):
            for j in range(len(trainX)):
                train[j][i]=trainX[j][caract[i]]
                
            for k in range(len(testX)):
                test[k][i]=testX[k][caract[i]]
        
        trainYNuevo=np.reshape(trainYNuevo, (len(trainY), -1))
        
        clf.fit(train, trainYNuevo)
        prediccion=clf.predict(test)
        
        mae+=metrics.mean_absolute_error(testY, prediccion)
        mse+=metrics.mean_squared_error(testY, prediccion)
        r2+=metrics.r2_score(testY, prediccion)
    
    print str("\nAprendizaje realizado con los atributos: ")+str(caract)
    print 'Error abs: ', mae/len(kf), 'Error cuadratico: ', mse/len(kf), 'R cuadrado: ', r2/len(kf)

#FIN FUNCION aprendizajePorCaract

def gradientBoosting():
    
    num_estimadores = 350
    clf = ensemble.GradientBoostingRegressor(n_estimators=num_estimadores, max_depth=2, learning_rate=0.1, loss='ls', subsample=0.5)
    
    importancias = [0,0,0,0,0,0,0,0,0,0,0,0,0]    
    mae, mse, mr2, cont = 0, 0, 0, 0
    test_score = np.zeros((num_estimadores,), dtype=np.float64)
    train_score = np.zeros((num_estimadores,), dtype=np.float64)
    mseVector = [0]

    kf = KFold(len(boston_Y), n_folds=10, indices=True)
    for train, test in kf:
        trainX, testX, trainY, testY=boston_X[train], boston_X[test], boston_Y[train], boston_Y[test]    
    
        clf.fit(trainX, trainY)
        pred = clf.predict(testX)
        
        maeGradient = metrics.mean_absolute_error(testY, pred)
        mseGradient = metrics.mean_squared_error(testY, pred)
        r2 = metrics.r2_score(testY, pred)
        
        mae = mae + maeGradient
        mse = mse + mseGradient     
        mr2 = mr2 + r2
        mseVector.append(mseGradient)
        cont = cont + 1
                
        for i, y_pred in enumerate(clf.staged_decision_function(testX)):
            test_score[i] = test_score[i] + clf.loss_(testY, y_pred)
         
        for i in range(num_estimadores):
            train_score[i] = clf.train_score_[i] + train_score[i]
    
        feature_importance = clf.feature_importances_
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        for i in range(13):
            importancias[i] = importancias[i] + feature_importance[i]

        print str("Iteracción ")+str(cont)+str(" de la validacion cruzada")
        print str("\tError medio absoluto:  ")+str(maeGradient)
        print str("\tError medio cuadrado:  ")+str(mseGradient)
        print str("\tr2:  ")+str(r2)

        #Dibuja los puntos que predice sobre los puntos verdaderos
        pl.plot(testY, testY, label='Valor verdadero')
        pl.plot(testY, pred, 'ro', label='Prediccion Gradient')
        pl.legend(bbox_to_anchor=(1.05, 1), borderaxespad=0., prop = FontProperties(size='smaller'))
        pl.show()

    print mseVector
    mae = mae/10
    mse = mse/10
    mr2 = mr2/10
    print str("Error medio absoluto: ")+str(mae)+str("\tError medio cuadratico: ")+str(mse)+str("\tR2: ")+str(mr2)    
    
    for i in range(13):
        importancias[i] = importancias[i]/10
        
    sorted_idx = np.argsort(importancias)
    pos = np.arange(sorted_idx.shape[0]) + .5
    importancias = np.reshape(importancias, (len(importancias), -1))

    boston = datasets.load_boston()
    pl.barh(pos, importancias[sorted_idx], align='center')
    pl.yticks(pos, boston.feature_names[sorted_idx])
    pl.xlabel('Importancia relativa')
    pl.show()
    
    for i in range(num_estimadores):
        test_score[i] = test_score[i]/10
        train_score[i] = train_score[i]/10
        
    pl.figure(figsize=(12, 6))
    pl.subplot(1, 1, 1)
    pl.title('Desviacion')
    pl.plot(np.arange(num_estimadores) + 1, train_score, 'b-', label='Error en el conjunto de Training')
    pl.plot(np.arange(num_estimadores) + 1, test_score, 'r-', label='Error en el conjunto de Test')
    pl.legend(loc='upper right')
    pl.xlabel('Iteracciones del Boosting (numero de arboles)')
    pl.ylabel('Desviacion')
    pl.show()
    
    print len(mseVector)
    print len(np.arange(10))    
    
    pl.subplot(1, 1, 1)
    pl.plot(np.arange(11), mseVector, 'b-')
    pl.legend(loc='upper right')
    pl.xlabel('Iteraccion de la validacion cruzada')
    pl.ylabel('Erro Medio Cuadratico')
    pl.show()

    
    fig, axs = plot_partial_dependence(clf, trainX,[0,1,2,3,4,5,6,7,8,9,10,11,12])
    
    fig.suptitle('Dependencia parcial del valor de las casas')
    
    pl.subplots_adjust(top=0.9)
        
    pl.show()
#FIN FUNCION gradientBoosting

def arbolesRegresion(caract):
    
    clf = DecisionTreeRegressor(min_samples_leaf=10, min_samples_split=15, max_depth=13, compute_importances=True)
    
    importancias = [0,0,0,0,0,0,0,0,0,0,0,0,0]    
    mae=mse=r2=0
    
    kf = KFold(len(boston_Y), n_folds=10, indices=True)
    for train, test in kf:
        trainX, testX, trainY, testY=boston_X[train], boston_X[test], boston_Y[train], boston_Y[test]
            
        nCar=len(caract)
        train=np.zeros((len(trainX), nCar))
        test=np.zeros((len(testX), nCar))
        trainYNuevo=trainY
        
        for i in range(nCar):
            for j in range(len(trainX)):
                train[j][i]=trainX[j][caract[i]]
                
            for k in range(len(testX)):
                test[k][i]=testX[k][caract[i]]
        
        trainYNuevo=np.reshape(trainYNuevo, (len(trainY), -1))
        
        clf.fit(train, trainYNuevo)
        prediccion=clf.predict(test)            
        
#        clf.fit(trainX, trainY)
#        prediccion=clf.predict(testX)
            
        mae+=metrics.mean_absolute_error(testY, prediccion)
        mse+=metrics.mean_squared_error(testY, prediccion)
        r2+=metrics.r2_score(testY, prediccion)
        
        feature_importance = clf.feature_importances_
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        for i in range(13):
            importancias[i] = importancias[i] + feature_importance[i]
        
    print 'Error abs: ', mae/len(kf), 'Error cuadratico: ', mse/len(kf), 'R cuadrado: ', r2/len(kf)
    
    for i in range(13):
        importancias[i] = importancias[i]/10
        
    sorted_idx = np.argsort(importancias)
    pos = np.arange(sorted_idx.shape[0]) + .5
    importancias = np.reshape(importancias, (len(importancias), -1))

    boston = datasets.load_boston()
    pl.barh(pos, importancias[sorted_idx], align='center')
    pl.yticks(pos, boston.feature_names[sorted_idx])
    pl.xlabel('Importancia relativa')
    pl.show()    
    
    import StringIO, pydot 
    dot_data = StringIO.StringIO() 
    tree.export_graphviz(clf, out_file=dot_data) 
    graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
    graph.write_pdf("bostonTree.pdf") 


def pruebaModelos():

    gra = ensemble.GradientBoostingRegressor(n_estimators=350, max_depth=2, learning_rate=0.1, loss='ls', subsample=0.5)
    svr = SVR(kernel='linear', C=0.1, epsilon=0.2)
    reg = LinearRegression()
    dtr = DecisionTreeRegressor(min_samples_leaf=10, min_samples_split=15, max_depth=13, compute_importances=True)
    
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
    
    parametros = [{'kernel':'linear', 'C':0.1, 'epsilon':0.2},
                  {'kernel':'linear', 'C':1.0, 'epsilon':0.2},
                  {'kernel':'rbf', 'degree':3, 'gamma':.0001, 'C':1.0, 'epsilon':0.2},
                  {'kernel':'rbf', 'degree':2, 'gamma':.01, 'C':0.1, 'epsilon':0.2}]

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
        
            print clf.coef_
        print "Parametros: ", c
        print 'Error abs: ', mae/len(kf), 'Error cuadratico: ', mse/len(kf), 'R cuadrado: ', r2/len(kf)
        mae=mse=r2=0



def ajustesParametrosG():
    parametros_gradient = [
                            #{'loss':'ls'},{'loss':'lad'}, {'loss':'huber'},
                           #{'n_estimators': 500, 'loss':'ls'},
                           {'n_estimators': 350, 'max_depth':2, 'learning_rate': 0.1, 'loss': 'ls', 'subsample':0.5},
                           {'n_estimators': 350, 'max_depth':2, 'learning_rate': 0.1, 'loss': 'ls'},
                           {'n_estimators': 500, 'max_depth': 2, 'min_samples_split': 4, 'min_samples_leaf':1, 'learning_rate': 0.01, 'loss': 'ls'},
                           {'n_estimators': 500, 'max_depth': 2, 'min_samples_split': 5, 'min_samples_leaf':1, 'learning_rate': 0.1, 'loss': 'ls'}]


    num_estimadores = 350
    test_score = np.zeros((num_estimadores,), dtype=np.float64)
    train_score = np.zeros((num_estimadores,), dtype=np.float64)

    for c in parametros_gradient:
        #print c
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
            
            for i, y_pred in enumerate(clf.staged_decision_function(testX)):
                test_score[i] = test_score[i] + clf.loss_(testY, y_pred)
         
            for i in range(num_estimadores):
                train_score[i] = clf.train_score_[i] + train_score[i]            
            
            
        print 'Error abs: ', mae/len(kf), 'Error cuadratico: ', mse/len(kf), 'R cuadrado: ', r2/len(kf)

        for i in range(num_estimadores):
            test_score[i] = test_score[i]/10
            train_score[i] = train_score[i]/10
        
        pl.figure(figsize=(12, 6))
        pl.subplot(1, 1, 1)
        pl.title('Desviacion')
        pl.plot(np.arange(num_estimadores) + 1, train_score, 'b-', label='Error en el conjunto de Training')
        pl.plot(np.arange(num_estimadores) + 1, test_score, 'r-', label='Error en el conjunto de Test')
        pl.legend(loc='upper right')
        pl.xlabel('Iteracciones del Boosting (numero de arboles)')
        pl.ylabel('Desviacion')
        pl.show()




#####################################
#                                   #
#   Llamadas a las funciones        #
#                                   #
#####################################

boston=datasets.load_boston()
boston_X= boston.data
boston_Y= boston.target

#normalizacion()
#pruebaModelos()
#aprendizajePorCaractGradient([0,2,3,4,5,6,7,8,9,10,11,12])
#gradientBoosting()
#arbolesRegresion([0,1,2,3,4,5,6,7,8,9,10,11,12])
#tecnicaSVR()
#ajustesParametrosG()
