# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 13:58:13 2019

@author: A669593
"""


from datetime import datetime
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from pandas.api.types import CategoricalDtype
from sklearn.preprocessing import MinMaxScaler,StandardScaler

import matplotlib.pyplot as plt
import json
import pickle

def getDFDesc(df):
	dfDesc=df.describe(include='all')
	dfDescT=dfDesc.T
	dfDescT['fillCount']=dfDescT['count']/len(df)*100
	if 'unique' in dfDescT.columns:
		dfDescT['distCount']=dfDescT['unique']/dfDescT['count']*100
	return dfDescT


def GetTimeStamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def PickleRead(filename):
    return pickle.load(open( filename, "rb" ))

def PickleWrite(obj,filename):
    pickle.dump(obj, open( filename, "wb" ) )

def ReadInputs(filename):
    return PickleRead(filename)

def WriteOutputs(obj,filename):
    PickleWrite(obj, filename)
	
from enum import Enum
class ImputeMethodEnum(Enum):
    ZERO=1
    MEAN=2
    MEDIAN=3
    MODE=4


def getDtypes(pdSeries):
    dTypes=  pdSeries.reset_index()
    dTypes.columns=['ColName','DType']
    return dTypes


def missing_value(df,misDict=None,dfDescT=None,imputeMethod=ImputeMethodEnum.MEDIAN):
    
    if misDict==None:
        misDict={}
    if dfDescT is None:
        dfDescT=df.describe(include='all').T
    
    
    dfDescT=dfDescT[dfDescT['count']<len(df)]    
        
    dftypes=getDtypes(df[list(dfDescT.index)].dtypes)
    for id,row in dftypes.iterrows():
        col=row['ColName']
        dtype=row['DType']
        if col not in misDict.keys():
            if dtype in ('int64','float64'):
                if imputeMethod==ImputeMethodEnum.MEAN:
                    misDict.update({col:df[col].mean()})
                    print(col," imputed by MEAN")
                elif imputeMethod==ImputeMethodEnum.MEDIAN:
                    misDict.update({col:df[col].median()})
                    print(col," imputed by MEDIAN")
                else:
                    misDict[col]=0
                    print(col," imputed by ZERO")
            else:
                misDict.update({col:dfDescT['top'][col]})
                print(col," imputed by MODE ",dfDescT['top'][col])
        
        df.loc[:,col].fillna(misDict[col], inplace=True)
            
    print('Missing value imputation done : ',imputeMethod)
    return df,misDict


def categorizeCols(df,cols,mapCat={}):
    print("CATEGORIES ENCODING")
    print("----------------------------")

    for col in cols:
        if col in mapCat.keys():
            cat_type = CategoricalDtype(categories=mapCat[col], ordered=True)
            df[col] = df[col].astype(cat_type)
            print("---------Category Order for col : ",col," from config------")
        else:
            df[col] = df[col].astype('category')
        #print(col, list(df[col].cat.categories))
    print("----------------------------")
    return df

def LabelEncodeCols(df,categorical_columns,onehotColumns): 
    
    
    
    if len(categorical_columns) >0:
        df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
        
    if len(onehotColumns) >0:
        df=pd.get_dummies(df, columns=onehotColumns)
    return df

def ChkZeroOne(col):
    res=col[col.isin([0,1]) == False]
    return len(res)==0


def normalize(df,features_to_normalize,scaler=None):
    """df[features_to_normalize] = df[features_to_normalize].apply(lambda x:(x-x.min()) / (x.max()-x.min()))
    return df"""
    if scaler is None:
        scaler = StandardScaler()
        df.loc[:,features_to_normalize] = scaler.fit_transform(df.loc[:,features_to_normalize])
        return df,scaler
    else:
        df.loc[:,features_to_normalize] = scaler.transform(df.loc[:,features_to_normalize])
        return df
    """
    for feature_name in cols:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        df[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return df"""


def trn_tst_split(X, y, split_percent,stratifyCol=None,random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=split_percent, random_state=random_state, stratify=stratifyCol)
    X_trainIndex=list(X_train.index)   
    X_testIndex=list(X_test.index)   
	   
    X_train.reset_index(inplace=True,drop=True)
    X_test.reset_index(inplace=True,drop=True)
    y_train.reset_index(inplace=True,drop=True)
    y_test.reset_index(inplace=True,drop=True)
    
    return X_train, X_test, y_train, y_test,X_trainIndex,X_testIndex
    

def GetkFoldedTrainVal(X,y,pred_variable_type,nFolds=5,randomstate=300):    
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import KFold

    if pred_variable_type == "categorical":
        print("using stratified fold")
        skf = StratifiedKFold(n_splits=nFolds,random_state=randomstate)
    else:
        skf = KFold(n_splits=nFolds,random_state=randomstate)
		
    
    for train_index, test_index in skf.split(X, y):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]
        X_train.reset_index(inplace=True,drop=True)
        X_test.reset_index(inplace=True,drop=True)
        y_train.reset_index(inplace=True,drop=True)
        y_test.reset_index(inplace=True,drop=True)
        print(y_train.value_counts())
        print(y_test.value_counts())		
        #X_train=np.array(X_train)
        #X_test=np.array(X_test)
		
        yield X_train, X_test ,y_train, y_test,train_index, test_index   
    
    
def GetScores(actualY,Pred,categoriesNames):
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import f1_score,recall_score
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.metrics import classification_report
    from sklearn.metrics import auc
    from sklearn.metrics import roc_auc_score
    from sklearn import metrics
	
    acc_score=accuracy_score(actualY, Pred,normalize=True)
    target_names = categoriesNames
    class_report=classification_report(actualY, Pred, target_names=target_names)
    f1score=f1_score(actualY, Pred)
    cnf_matrix = confusion_matrix(actualY, Pred)
    rsAllClass=recall_score(actualY, Pred, average=None) 
    rs=recall_score(actualY, Pred, average='weighted') 
    fpr, tpr, thresholds = metrics.roc_curve(actualY, Pred, pos_label=2)
    aucRes=metrics.auc(fpr, tpr) 
    roc_auc_scoreVal=roc_auc_score(actualY, Pred)
    return acc_score,rsAllClass,rs,f1score,cnf_matrix,class_report,aucRes,fpr, tpr,roc_auc_scoreVal    



def GetAdjustedPredictions(y_pred_p,scale_ratio_1,scale_ratio_2,threshold):
    num_vals = len(y_pred_p)
    default_prob = np.zeros(num_vals)
    y_pred = np.zeros(num_vals)
    for i in range(num_vals):
        #default_p[i] = y_pred_p[i][1]
        if(y_pred_p[i] < threshold):
            y_pred[i] = 0
            p = scale_ratio_1*y_pred_p[i]
        else:
            y_pred[i] = 1
            p = 1 - scale_ratio_2*(1 - y_pred_p[i])
        default_prob[i] = p
    return default_prob,y_pred
    
def getFeatureImp(res,numOfTopFeatures):
    featImportance={}
    feature_cover_perc=0
    fImp=pd.DataFrame(res.leader.varimp())
    
    if len(fImp)>0:
        fImp.columns=['variable','relative_importance','scaled_importance','percentage']
        topNFeatures=0
        for index,row in fImp.iterrows():
            #print(row)
            featImportance[row['variable']]=row['percentage']
            if index < numOfTopFeatures:
                topNFeatures+=row['percentage']
        
        feature_cover_perc=round(topNFeatures/sum(fImp['percentage']),4) 
        
    return featImportance,feature_cover_perc,fImp
    
def getCapturefeatureOutput(allscores,classInterest,y_test,Pred,res,numOfTopFeatures):
    captures=list(allscores[1])
 
    classCapture=captures.pop(classInterest)
    nonclassCapture=captures[0]
    
    classInterestCountActual=len(y_test[y_test==classInterest])
    classInterestCountPred=len(Pred[Pred==classInterest])
    accuracy=allscores[0]
    accuracy_per_roc_auc =  allscores[9]
    
    featImportance,feature_cover_perc,fImp=getFeatureImp(res,numOfTopFeatures)
    
    output_json_data = {"Default Capture": classCapture, "Non-Default Capture": nonclassCapture,
                        "True Default Percentage": classInterestCountActual,
                        "Predicted Suspicious Percentage":classInterestCountPred,
                        "Overall Accuracy":accuracy, "ROC Accuracy":accuracy_per_roc_auc,
                       "Features":featImportance,"top_5_feature_cover":feature_cover_perc}
    
    return output_json_data,fImp
    
    
def getOutliers(df):
	from pyod.models.abod import ABOD
	from pyod.models.knn import KNN
	
	outlier_fraction = 0.1
	outlierlist=[]
	
	classifiers = {
	     'Angle-based Outlier Detector (ABOD)'   : ABOD(contamination=outlier_fraction),
	     'K Nearest Neighbors (KNN)' :  KNN(contamination=outlier_fraction)
	}
	classifiers = {
	     'Angle-based Outlier Detector (ABOD)'   : ABOD(contamination=outlier_fraction),
	}
	print("outlier detection")
	for i, (clf_name, clf) in enumerate(classifiers.items()):
	    print(clf_name)
	    clf.fit(df)
	    # predict raw anomaly score
	    #scores_pred = clf.decision_function(df) * -1
		
	    y_pred = clf.predict(df)
	    #n_inliers = len(y_pred) - np.count_nonzero(y_pred)
	    n_outliers = np.count_nonzero(y_pred == 1)
	    print(clf_name,n_outliers)
	    outlierlist.append((n_outliers,y_pred.tolist()))  
		
	return outlierlist		  
    
    
def plot_corr(df,size=10):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''

    corr = df.corr()
    corr.style.background_gradient(cmap='coolwarm', axis=None)
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical')
    plt.yticks(range(len(corr.columns)), corr.columns)   
	
	
	
def GetClassWeights(y_train):
	from sklearn.utils import class_weight
	class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)
	return class_weights	
	