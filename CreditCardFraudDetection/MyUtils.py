# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 16:25:27 2018

@author: A669593
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def fnHandleMsg(msg,msgType=""):
    print(msg)


def fnAssignColnames(df,colNames=None):
    if colNames is not None and len(colNames)==len(df.columns):
        df.columns=colNames

    return df

    
    
def fnCreatePdFrameFromArray(arrayTuple,colNames=None):
    #return pd.DataFrame(np.column_stack(arrayTuple))
    df=None
    for a in arrayTuple:
        if len(a.shape)==1:
            n=pd.Series(a)
        else:
            n=pd.DataFrame(a)
        if df is None:
            df=pd.DataFrame(n)
        else:
            df=pd.concat([df,n],axis=1)

    return df

def fnGetDf(data,colNames=None):
    df=fnCreatePdFrameFromArray(data,colNames)
    df=fnAssignColnames(df,colNames)
    return df
                        
def fnFilterMissingValues(df):
    missingValues=df.isnull().sum(axis=1)
    missingValuesCount=sum(missingValues>0)
    fnHandleMsg("No. of Missing Values in DF : "+str(missingValuesCount))
    df=df.dropna()
    return df,missingValuesCount

def fnGetUniqueValLenColumns(df):
    iDfLen=len(df)
    lUniqueValCount=[]
    for col in df.columns:
        iUniqueValLen=len(np.unique(df[col]))
        lUniqueValCount.append(iUniqueValLen)
        
    return lUniqueValCount


def fnNormalizeCols(df,colNames):
    fnHandleMsg("Normalizing Values in DF for columns: "+str(colNames))
    from sklearn import preprocessing
    dfNew = pd.DataFrame(df[colNames])
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(dfNew)
    df_normalized = pd.DataFrame(np_scaled)
    df_normalized.columns=colNames
    df[colNames]=df_normalized
    #for col in colNames:
        #df[col]=df_normalized[col]
        
    return df

def fnPlotCols(df,colNames=None,num_bins=50):

    if colNames is None:
        colNames=df.columns
        
    for colName in colNames:
        colData=df[colName]
        fnPlotHistogram(colData,colName,num_bins)
        fnPlotBoxplot(colData,colName)
        

def fnPlotHistogram(colData,colName,num_bins=50):
    mu=np.mean(colData)
    sigma=np.std(colData)


    fig, ax = plt.subplots()

    # the histogram of the data
    n, bins, patches = ax.hist(colData.dropna(), num_bins,normed=True)

    # add a 'best fit' line
    y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
    ax.plot(bins, y, '--')
    ax.set_xlabel('Smarts')
    ax.set_ylabel('Probability density')
    ax.set_title(r'Histogram of {0}: mu={1}, sigma={2}'.format(colName,mu,sigma))

    # Tweak spacing to prevent clipping of ylabel
    #fig.tight_layout()
    plt.show()

def fnPlotBoxplot(colData,colName):
    #colData=df[colName]
    fig1, ax1 = plt.subplots()
    ax1.set_title(r'Box Plot of {0}'.format(colName))
    ax1.boxplot(colData.dropna())

    # Tweak spacing to prevent clipping of ylabel
    fig1.tight_layout()
    plt.show()

def plotAllHist(df):
	#Plot histograms for all columns
	fig = plt.figure(figsize=(20, 15))
	# loop over all vars (total: 34)
	for i in range(0, df.shape[1]):
	    plt.subplot(6, 6, i+1).set_title(df.columns[i])
	    #f = plt.gca()
	    #f.axes.get_yaxis().set_visible(False)
	    # f.axes.set_ylim([0, train.shape[0]])
	
	    vals = np.size(df.iloc[:, i].unique())
	    if vals < 10:
	        bins = vals
	    else:
	        vals = 10
			
	
	    plt.hist(df.iloc[:, i], bins=100, color='#3F5D7D')
	
	plt.tight_layout()
	
	plt.savefig("histogram-distribution.png")


    
def RandomizeDF(df,seed=300):    
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    df=df.loc[perm]
    df.reset_index(inplace=True)
    return df

def GetXyDf(df,outputVarName):
    y=df[outputVarName]
    X=df.loc[:, df.columns != outputVarName]
    return X,y
    
    
    
def GetTrainTest(df,outputVarName,testPer=0.2,seed=200,randomstate=300):
    from sklearn.model_selection import train_test_split
    X,y=GetXyDf(df,outputVarName)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y, 
                                                    test_size=testPer,random_state=randomstate)
    
    X_train.reset_index(inplace=True,drop=True)
    X_test.reset_index(inplace=True,drop=True)
    y_train.reset_index(inplace=True,drop=True)
    y_test.reset_index(inplace=True,drop=True)
    
    return X_train, X_test, y_train, y_test
    
    
    
def GetkFoldedTrainVal(X,y,nFolds=5,randomstate=300):    
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=nFolds,random_state=randomstate)
    
    for train_index, test_index in skf.split(X, y):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]
        yield X_train, X_test ,y_train, y_test
            
            
def GetScores(actualY,Pred,categoriesNames):
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import f1_score,recall_score
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.metrics import classification_report
    acc_score=accuracy_score(actualY, Pred,normalize=True)
    target_names = categoriesNames
    f1score=classification_report(actualY, Pred, target_names=target_names)
    cnf_matrix = confusion_matrix(actualY, Pred)
    rs=recall_score(actualY, Pred, average='weighted')  
    return acc_score,rs,f1score,cnf_matrix

def PlotRecallCurve(yactual,ypred):
	from sklearn.metrics import roc_curve,auc
	fpr, tpr, thresholds = roc_curve(yactual,ypred)
	roc_auc = auc(fpr,tpr)
	
	# Plot ROC
	plt.title('Receiver Operating Characteristic')
	plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
	plt.legend(loc='lower right')
	plt.plot([0,1],[0,1],'r--')
	plt.xlim([-0.1,1.0])
	plt.ylim([-0.1,1.01])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.show()