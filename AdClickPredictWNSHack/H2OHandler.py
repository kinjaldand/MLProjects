# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 16:04:45 2019

@author: A669593
"""


import h2o
"""from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator"""
from h2o.automl import H2OAutoML

def GetBestH2OModel(train,XCols,y_col,isCat,X_test,weights_column,stopping_metric):

    h2o.init()
    print(isCat)
    print(train[y_col].value_counts())	
    train = h2o.H2OFrame(train)
    if isCat==True:
        train[y_col] = train[y_col].asfactor()
    aml = H2OAutoML(max_models = 20, max_runtime_secs=100, seed = 1,stopping_metric=stopping_metric)
    aml.train(x=XCols, y=y_col, training_frame=train, weights_column = weights_column)
    predtrain=aml.leader.predict(train)
    predtrain=predtrain.as_data_frame()
    Pred=aml.leader.predict(h2o.H2OFrame(X_test))
    predDf=Pred.as_data_frame()
    ptrain=train[y_col]
    ptrain=ptrain.as_data_frame()
    #h2o.save_model(aml.leader, path = "./best_model_bin")
    return aml,predDf,predtrain,ptrain
    """
    lb = aml.leaderboard
    print(lb)
    
    metalearner = h2o.get_model(aml.leader.metalearner()['name'])
    metalearner.std_coef_plot()
    return metalearner"""
