# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 23:16:21 2019

@author: ADMIN
"""



import pandas as pd
import numpy as np
import AllFunctions as af
#import dateutil
import math

item_data=pd.read_csv("item_data.csv")
log_data=pd.read_csv("view_log.csv",parse_dates=['server_time'],infer_datetime_format=True)
train_data=pd.read_csv("train.csv",parse_dates=['impression_time'],infer_datetime_format=True)
test_data=pd.read_csv("test.csv",parse_dates=['impression_time'],infer_datetime_format=True)

test_data['is_click']=-1

train_test_data=pd.concat([train_data,test_data],axis=0)

ismall=item_data.head(5)
#dfDesc=af.getDFDesc(item_data)

item_data=af.categorizeCols(item_data,cols=['item_id','category_1', 'category_2', 'category_3','product_type'])

#dfDesc=af.getDFDesc(item_data)
item_data['item_price_log']=np.log(item_data['item_price'])

#dfDesc=af.getDFDesc(log_data)

log_data=af.categorizeCols(log_data,cols=['session_id','user_id','item_id'])

#dfDesc=af.getDFDesc(log_data)

log_item_data=pd.merge(log_data,item_data,on='item_id')

train_test_data['impression_time_7less'] = train_test_data['impression_time'] -  pd.to_timedelta(7, unit='d')


train_test_data.reset_index(inplace=True,drop=True)

"""user_ids_list =  np.unique(log_data['user_id'])

minimptime=np.min(train_test_data['impression_time_7less'])

l=log_item_data.sample(frac=0.001,random_state=5)
l.to_csv('lShort.csv',index=False)

t=train_test_data.sample(frac=0.01,random_state=5)
t.to_csv('tShort.csv',index=False)"""


log_item_data.to_csv('log_item_data.csv',index=False)
train_test_data.to_csv('train_test_data.csv',index=False)

def calcAllOutputs(df):
	visit_count = len(df)
	total_sessions = len(np.unique(df['session_id']))
	total_items = len(np.unique(df['item_id']))
	total_category_1 = len(np.unique(df['category_1']))
	total_category_2 = len(np.unique(df['category_2']))
	total_category_3 = len(np.unique(df['category_3']))
	total_product_type = len(np.unique(df['product_type']))
	
	item_price_max = np.max(df['item_price_log'])
	item_price_min = np.min(df['item_price_log'])
	item_price_avg = np.mean(df['item_price_log'])
	item_price_std = np.std(df['item_price_log'])
	max_time=np.max(df['server_time'])
	impid=np.max(df['impression_id'])
	#diff = df['impression_time'] - max_time
	#diff1 = diff.total_seconds()
	res=[impid,visit_count,total_sessions ,total_items ,total_category_1 ,total_category_2 ,total_category_3 ,total_product_type ,item_price_max ,item_price_min ,item_price_avg ,item_price_std ,max_time]
	return res
	
def calcImpFeatures(df):
	previous_imp_app_count=len(np.unique(df['app_code']))
	max_time2=np.max(df['impression_time_y'])	
	impid=np.max(df['impression_id'])
	return [impid,max_time2,previous_imp_app_count]

def calcAppFeatures(df):
	previous_imp_same_app_count=len(np.unique(df['app_code']))
	max_time3=np.max(df['impression_time_y'])	
	impid=np.max(df['impression_id'])
	return [impid,max_time3,previous_imp_same_app_count]

def applymathfloor(x):
	if np.isnan(x)==False:
		return math.floor(x)
	else:
		return x

dfC=train_test_data.merge(log_item_data,on='user_id')
print(len(dfC))
dfC2 = dfC[(dfC.server_time < dfC.impression_time)]
print(len(dfC2))
dfCHead=dfC2.head(100)


dfC3=dfC2.groupby('impression_id').apply(calcAllOutputs)

dfFeatureset1=pd.DataFrame.from_records(dfC3)
dfFeatureset1.columns=['impression_id','visit_count','total_sessions','total_items','total_category_1','total_category_2','total_category_3','total_product_type','item_price_max ','item_price_min','item_price_avg','item_price_std','max_time']
dfFeatureset1.to_csv('dfFeatureset1.csv',index=False)

dfC=train_test_data.merge(train_test_data[['user_id','impression_time','app_code']],on='user_id',suffixes=('', '_y'))
dfC2=dfC[dfC.impression_time<dfC.impression_time_y]
dfC3=dfC2.groupby('impression_id').apply(calcImpFeatures)
dfFeatureset2=pd.DataFrame.from_records(dfC3)
dfFeatureset2.columns=['impression_id','max_time2','previous_imp_app_count']
dfFeatureset2.to_csv('dfFeatureset2.csv',index=False)

dfC4=dfC2[dfC2.app_code==dfC2.app_code_y]
dfC5=dfC4.groupby('impression_id').apply(calcAppFeatures)
dfFeatureset3=pd.DataFrame.from_records(dfC5)
dfFeatureset3.columns=['impression_id','max_time3','previous_imp_same_app_count']
dfFeatureset3.to_csv('dfFeatureset3.csv',index=False)

"""
train_test_data=pd.read_csv('train_test_data.csv',parse_dates=['impression_time'],infer_datetime_format=True)
dfFeatureset1=pd.read_csv('dfFeatureset1.csv',parse_dates=['max_time'],infer_datetime_format=True)
dfFeatureset2=pd.read_csv('dfFeatureset2.csv',parse_dates=['max_time2'],infer_datetime_format=True)
dfFeatureset3=pd.read_csv('dfFeatureset3.csv',parse_dates=['max_time3'],infer_datetime_format=True)

"""
mergeddf=train_test_data.merge(dfFeatureset1,on='impression_id',how='left')
mergeddf=mergeddf.merge(dfFeatureset2,on='impression_id',how='left')
mergeddf=mergeddf.merge(dfFeatureset3,on='impression_id',how='left')


mergeddf['diff1']=(mergeddf['impression_time']-mergeddf['max_time']).dt.total_seconds()
mergeddf['diff2']=(mergeddf['max_time2']-mergeddf['impression_time']).dt.total_seconds()
mergeddf['diff3']=(mergeddf['max_time3']-mergeddf['impression_time']).dt.total_seconds()

train_test_data=mergeddf

s=train_test_data.app_code.value_counts()
s=s/len(train_test_data)
train_test_data['app_id']=train_test_data['app_code'].apply(lambda x: s[x])


train_test_data['diff_days']=(train_test_data['diff1']/3600/24).apply(applymathfloor)
#train_test_data['diff_hours']=(train_test_data['diff1']/3600).apply(applymathfloor)
#train_test_data['diff_mins']=(train_test_data['diff1']/60).apply(applymathfloor)
#train_test_data['diff_secs']=(train_test_data['diff1']).apply(applymathfloor)

#train_test_data['prev_diff_days']=(train_test_data['diff2']/3600/24).apply(applymathfloor)
train_test_data['prev_diff_hours']=(train_test_data['diff2']/3600).apply(applymathfloor)
#train_test_data['prev_diff_mins']=(train_test_data['diff2']/60).apply(applymathfloor)
#train_test_data['prev_diff_secs']=(train_test_data['diff2']).apply(applymathfloor)
#train_test_data['prev_app_diff_days']=(train_test_data['diff3']/3600/24).apply(applymathfloor)
train_test_data['prev_app_diff_hours']=(train_test_data['diff3']/3600).apply(applymathfloor)
#train_test_data['prev_app_diff_mins']=(train_test_data['diff3']/60).apply(applymathfloor)
#train_test_data['prev_app_diff_secs']=(train_test_data['diff3']).apply(applymathfloor)

train_test_data['it_day_of_week'] = train_test_data['impression_time'].dt.dayofweek
train_test_data['it_month_start'] = train_test_data['impression_time'].dt.is_month_start
train_test_data['it_month_end'] = train_test_data['impression_time'].dt.is_month_end
train_test_data['it_weekday'] = train_test_data['impression_time'].apply(lambda x: x.weekday())


train_test_data=train_test_data.drop(columns=['impression_id','impression_time','user_id','impression_time_7less','app_code','max_time','max_time2','max_time3'])	
train_test_data=train_test_data.drop(columns=['diff1','diff2','diff3'])
train_test_data=train_test_data.fillna(0)
train_test_data=af.categorizeCols(train_test_data,cols=['os_version','it_day_of_week','it_weekday'])
train_test_data=af.LabelEncodeCols(train_test_data.copy(),onehotColumns=[], categorical_columns=['os_version','it_day_of_week','it_weekday'])	

train_test_data.to_csv("train_test_dataAll.csv",index=False)


X=train_test_data 
#af.plot_corr(X)
#X=X.drop(columns=[x  for x in dfFeatureset1.columns if x in X.columns])   
X=X.drop(columns=['previous_imp_app_count','prev_app_diff_hours'])
X=X.drop(columns=['total_category_2','total_category_3','total_sessions', 'item_price_min','item_price_max ','item_price_std'])
X=X.drop(columns=['total_product_type','visit_count'])
print(X.columns)
pred_variable_type = "categorical"
target_variable='is_click'
TrainCleanVars={}
    
X_trainVal=X[X['is_click']!= -1]
X_test=X[X['is_click']== -1]
X_test=X_test.drop(columns=['is_click'])


X_trainVal.reset_index(inplace=True,drop=True)
X_test.reset_index(inplace=True,drop=True)

zeroOneCols=X_trainVal.apply(lambda x: af.ChkZeroOne(x))
standarizeCols=list(zeroOneCols[zeroOneCols==False].index)

X_trainVal,scaler=af.normalize(X_trainVal,standarizeCols)
#standarizeCols.remove(target_variable)
X_test=af.normalize(X_test,standarizeCols,scaler)

trainVal_frame=X_trainVal




x_cols=list(X_trainVal.columns)
y_col=target_variable
trainVal_frame[target_variable] = trainVal_frame[target_variable].astype(np.uint8)
	
import H2OHandler as hh 

print("Start H2O model training")

#H2o internally uses k-fold cross validation
res,PredDF,predtrain=hh.GetBestH2OModel(trainVal_frame,x_cols,y_col,pred_variable_type == "categorical",X_test)
allscores=af.GetScores(train_data[target_variable],predtrain['predict'],['NO','YES'])
#Best model scores on entire available train set
print(allscores[4],allscores[9])
TrainCleanVars['H2OBestModel']=res.leader
X_test[target_variable]=PredDF['predict']
X_test[standarizeCols]=scaler.inverse_transform(X_test[standarizeCols])

ts=af.GetTimeStamp()
af.PickleWrite(TrainCleanVars,"TrainCleanVars"+str(ts)+".pkl")


X_test['impression_id']=test_data['impression_id']
final_sub=X_test[['impression_id',target_variable]]
final_sub.to_csv('samplesubmission'+str(ts)+'.csv',index=False)

lb=res.leaderboard
lbres=lb[:5,"model_id"]
import h2o

m = h2o.get_model(lb[0,"model_id"])
varimpres=m.varimp(use_pandas=True)




