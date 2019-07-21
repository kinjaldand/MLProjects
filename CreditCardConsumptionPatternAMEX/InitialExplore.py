# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 13:34:12 2019

@author: ADMIN
"""


import AllFunctions as af
import pandas as pd
import numpy as np
import pandas_profiling
#import H2OHandler as hh 
df=pd.read_csv('train.csv')
orgCC = df['cc_cons'].copy()
df['isTrain']=True
df2=pd.read_csv('test.csv')
df2['isTrain']=False
df2['cc_cons']=-1

fillCountMinPer = 50
idCols=['id']
distCatPer=2
onehotColumns=[]
pred_variable_type='regression'
target_variable = 'cc_cons'

TrainCleanVars={}
TrainCleanVars['dropCols']=[]


# Account Desc
dfDescT=af.getDFDesc(df)

dfDescT2=af.getDFDesc(df2)

df3=pd.concat([df,df2],ignore_index=True)
dfDescT3=af.getDFDesc(df3)
df=df3.reset_index(drop=True)

dfDescT=af.getDFDesc(df)

#profile = df.profile_report(title='Pandas Profiling Report')
#profile.to_file(output_file="output.html")

#rejected_variables = profile.get_rejected_variables(threshold=0.9)

#age has some unusual values like 224 which are quite invalid hence we will trim all such values to 75
df.loc[df['age'] > 75, 'age'] = 75

#Many amount columns are skewed lets take log and profile the results

cols=['card_lim',  'cc_cons_apr',
       'cc_cons_jun', 'cc_cons_may', 'cc_count_apr', 'cc_count_jun',
       'cc_count_may', 'credit_amount_apr', 'credit_amount_jun',
       'credit_amount_may', 'credit_count_apr', 'credit_count_jun',
       'credit_count_may', 'dc_cons_apr', 'dc_cons_jun', 'dc_cons_may',
       'dc_count_apr', 'dc_count_jun', 'dc_count_may', 'debit_amount_apr',
       'debit_amount_jun', 'debit_amount_may', 'debit_count_apr',
       'debit_count_jun', 'debit_count_may', 'emi_active',
        'max_credit_amount_apr', 'max_credit_amount_jun',
       'max_credit_amount_may']

#for col in cols:
#	df[col]=np.log(df[col]+1)
	
#profile = df.profile_report(title='Pandas Profiling Report after amount log')
#profile.to_file(output_file="output_log.html")	


"""import matplotlib.pyplot as plt

plt.matshow(df.corr())
plt.show()

f = plt.figure(figsize=(19, 15))
plt.matshow(df.corr(), fignum=f.number)
plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=45)
plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16)
f.savefig('CorrMatrix.png')
"""



"""columns=['personal_loan_active','personal_loan_closed','vehicle_loan_active', 'vehicle_loan_closed','investment_1', 'investment_2', 'investment_3', 'investment_4']
df[columns]=df[columns].fillna(0)

df['loan_enq']=df['loan_enq'].fillna('N')
dfDescT=af.getDFDesc(df)"""

TrainCleanVars['dropCols'].extend(idCols)
df.drop(columns=idCols,inplace=True)
print("Dropping cols as declared as id cols in config : ",idCols)



#Missing Value Imputation
# Now here many columns have missing values especially debit ones related, we have to fill them using data dictionary





df['cc_cons_highest'] = df[['cc_cons_apr','cc_cons_may','cc_cons_jun']].max(axis=1)
df['cc_cons_lowest'] = df[['cc_cons_apr','cc_cons_may','cc_cons_jun']].min(axis=1)
df['cc_cons_total'] = df[['cc_cons_apr','cc_cons_may','cc_cons_jun']].sum(axis=1)
df['cc_cons_average'] = df[['cc_cons_apr','cc_cons_may','cc_cons_jun']].mean(axis=1)
df['cc_cons_trans_avg']=df['cc_cons_total']/df[['cc_count_apr','cc_count_may','cc_count_jun']].sum(axis=1)
df['cc_cons_high_low_range']=df['cc_cons_highest']-df['cc_cons_lowest']
df['cc_cons_limit_crossed']=df['cc_cons_highest']>df['card_lim']
df['cc_cons_total_lim_ratio']=(df['cc_cons_total']/3)/df['card_lim']

"""df['dc_cons_highest'] = df[['dc_cons_apr','dc_cons_may','dc_cons_jun']].max(axis=1)
df['dc_cons_lowest'] = df[['dc_cons_apr','dc_cons_may','dc_cons_jun']].min(axis=1)
df['dc_cons_total'] = df[['dc_cons_apr','dc_cons_may','dc_cons_jun']].sum(axis=1)
df['dc_cons_average'] = df[['dc_cons_apr','dc_cons_may','dc_cons_jun']].mean(axis=1)
df['dc_cons_trans_avg']=df['dc_cons_total']/df[['dc_count_apr','dc_count_may','dc_count_jun']].sum(axis=1)
df['dc_cons_high_low_range']=df['dc_cons_highest']-df['dc_cons_lowest']

df['debit_amount_highest'] = df[['debit_amount_apr','debit_amount_may','debit_amount_jun']].max(axis=1)
df['debit_amount_lowest'] = df[['debit_amount_apr','debit_amount_may','debit_amount_jun']].min(axis=1)
df['debit_amount_total'] = df[['debit_amount_apr','debit_amount_may','debit_amount_jun']].sum(axis=1)
df['debit_amount_average'] = df[['debit_amount_apr','debit_amount_may','debit_amount_jun']].mean(axis=1)
df['debit_amount_trans_avg']=df['debit_amount_total']/df[['dc_count_apr','dc_count_may','dc_count_jun']].sum(axis=1)
df['debit_amount_high_low_range']=df['debit_amount_highest']-df['debit_amount_lowest']

df['credit_amount_highest'] = df[['credit_amount_apr','credit_amount_may','credit_amount_jun']].max(axis=1)
df['credit_amount_lowest'] = df[['credit_amount_apr','credit_amount_may','credit_amount_jun']].min(axis=1)
df['credit_amount_total'] = df[['credit_amount_apr','credit_amount_may','credit_amount_jun']].sum(axis=1)
df['credit_amount_average'] = df[['credit_amount_apr','credit_amount_may','credit_amount_jun']].mean(axis=1)
df['credit_amount_trans_avg']=df['credit_amount_total']/df[['dc_count_apr','dc_count_may','dc_count_jun']].sum(axis=1)
df['credit_amount_high_low_range']=df['credit_amount_highest']-df['credit_amount_lowest']

df['max_credit_amount_highest'] = df[['max_credit_amount_apr','max_credit_amount_may','max_credit_amount_jun']].max(axis=1)
df['max_credit_amount_lowest'] = df[['max_credit_amount_apr','max_credit_amount_may','max_credit_amount_jun']].min(axis=1)
df['max_credit_amount_total'] = df[['max_credit_amount_apr','max_credit_amount_may','max_credit_amount_jun']].sum(axis=1)
df['max_credit_amount_average'] = df[['max_credit_amount_apr','max_credit_amount_may','max_credit_amount_jun']].mean(axis=1)
df['max_credit_amount_trans_avg']=df['max_credit_amount_total']/df[['dc_count_apr','dc_count_may','dc_count_jun']].sum(axis=1)
df['max_credit_amount_high_low_range']=df['max_credit_amount_highest']-df['max_credit_amount_lowest']


df['cc_dc_cons_ratio'] = df['cc_cons_total'] / df['dc_cons_total']
df['credit_debit_ratio'] = df['credit_amount_total'] / df['debit_amount_total']
df['dc_count_total']=df[['dc_count_apr','dc_count_may','dc_count_jun']].sum(axis=1)
df['cc_count_total']=df[['cc_count_apr','cc_count_may','cc_count_jun']].sum(axis=1)
df['cc_dc_count_ratio']=df['cc_count_total']/df['cc_count_total']"""

df=df.replace([np.inf, -np.inf], np.nan)

dfDescT=af.getDFDesc(df)

#lets drop cols which are less than minimum fillcount for now. We can later revisit them if required
dropFlag=dfDescT[(dfDescT['fillCount']<fillCountMinPer) | (dfDescT['unique']==1) |  (dfDescT['std']==0)]
dropCols=list(dropFlag.index)
TrainCleanVars['dropCols'].extend(dropCols)
print("Dropping cols as unique count less or fillcount less or std is zero : ",dropCols)
df.drop(columns=dropCols,inplace=True)

df.to_csv('AfterFeature.csv',index=False)

"""plt.subplot(1, 2, 1)
plt.scatter(df['cc_cons_highest'],df['card_lim'],c="b")
plt.xlabel("highest spend")
plt.ylabel("card_lim")
plt.subplot(1, 2, 2)
plt.scatter(df['cc_cons_lowest'],df['card_lim'],c="r")
plt.xlabel("lowest spend")
plt.ylabel("card_lim")
plt.show()

div_val=10000
sc=plt.scatter(df['cc_cons_lowest']/div_val,df['cc_cons_highest']/div_val,c=df['card_lim']/div_val)
plt.colorbar(sc)
plt.xlabel("lowest spend")
plt.ylabel("highest spend")
plt.show()

plt.scatter(df['cc_cons_highest']/div_val,df['card_lim']/div_val,c=df['cc_cons_limit_crossed'])
plt.xlabel("highest spend")
plt.ylabel("card_lim")

plt.hist(df['card_lim'].dropna())
plt.show()

plt.hist(np.log(df.loc[df['isTrain']==True,'cc_cons']+1))
plt.show()
"""
#df.loc[df['isTrain']==True,'cc_cons']=np.log(df.loc[df['isTrain']==True,'cc_cons']+1)

dfDescT=af.getDFDesc(df)
catFlag=dfDescT[(dfDescT['distCount']<=distCatPer)]
catCols=list(catFlag.index)
df=af.categorizeCols(df,catCols)

catCols=list(set(catCols)-set(onehotColumns))
df=af.LabelEncodeCols(df.copy(),catCols,onehotColumns)

zeroOneCols=df.apply(lambda x: af.ChkZeroOne(x))
standarizeCols=list(zeroOneCols[zeroOneCols==False].index)

#standarizeCols.remove(target_variable)
"""profile = df.profile_report(title='Pandas Profiling Report')
profile.to_file(output_file="outputFeature.html")

rejected_variables = profile.get_rejected_variables(threshold=0.9)
df.drop(columns=rejected_variables,inplace=True)


standarizeCols = list(set(standarizeCols) - set(rejected_variables))
"""



X=df	    
    
X_trainVal=X[X['isTrain']==True]
X_test=X[X['isTrain']==False]


X_trainVal.reset_index(inplace=True,drop=True)
X_test.reset_index(inplace=True,drop=True)

X_trainVal.drop(columns=['isTrain'],inplace=True)
X_test.drop(columns=['isTrain'],inplace=True)



X_trainVal,misDict=af.missing_value(X_trainVal)
X_test,_=af.missing_value(X_test,misDict=misDict)

outlierlist=af.getOutliers(X_trainVal)

y_pred_outliers=np.array(outlierlist[0][1])

df_outliers=X_trainVal[y_pred_outliers==1]
dfDescT=af.getDFDesc(df_outliers)
X_trainVal=X_trainVal[y_pred_outliers==0]
dfDescT2=af.getDFDesc(X_trainVal)

X_trainVal,scaler=af.normalize(X_trainVal,standarizeCols)
#standarizeCols.remove(target_variable)
X_test=af.normalize(X_test,standarizeCols,scaler)
X_test.drop(columns=[target_variable],inplace=True)

dfDesc=X_test.describe(include='all')
dfDescT=dfDesc.T

trainVal_frame=X_trainVal

x_cols=list(X_trainVal.columns)
y_col=target_variable
	
import H2OHandler as hh 

print("Start H2O model training")


res,PredDF,predtrain=hh.GetBestH2OModel(trainVal_frame,x_cols,y_col,pred_variable_type == "categorical",X_test)
TrainCleanVars['H2OBestModel']=res.leader
X_test[target_variable]=PredDF['predict']
X_test[standarizeCols]=scaler.inverse_transform(X_test[standarizeCols])

ts=af.GetTimeStamp()
af.PickleWrite(TrainCleanVars,"TrainCleanVars"+str(ts)+".pkl")

X_test[X_test < 0]=0  #Need to fix this

X_test['id']=df2['id']
final_sub=X_test[['id',target_variable]]
final_sub.to_csv('samplesubmission'+str(ts)+'.csv',index=False)

lb=res.leaderboard
lbres=lb[:5,"model_id"]
import h2o

m = h2o.get_model(lb[0,"model_id"])
varimpres=m.varimp(use_pandas=True)


trainVal_frameCopy=trainVal_frame.copy()
trainVal_frameCopy.reset_index(inplace=True,drop=True)
trainVal_frameCopy['cc_cons']=predtrain
trainVal_frameCopy[standarizeCols]=scaler.inverse_transform(trainVal_frameCopy[standarizeCols])
trainVal_frameCopy[trainVal_frameCopy < 0]=0
orgCC=orgCC[y_pred_outliers==0]
trainVal_frameCopy['cc_cons_org']=orgCC
trainVal_frameCopy['diff']=trainVal_frameCopy['cc_cons_org']-trainVal_frameCopy['cc_cons']
trainCompare=trainVal_frameCopy[['cc_cons_org','cc_cons','diff']]

from sklearn.metrics import mean_squared_log_error
rmsle=np.sqrt(mean_squared_log_error(orgCC, trainVal_frameCopy['cc_cons']))
print(rmsle)


