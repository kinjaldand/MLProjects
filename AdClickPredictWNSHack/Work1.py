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
dfDesc=af.getDFDesc(item_data)

item_data=af.categorizeCols(item_data,cols=['item_id','category_1', 'category_2', 'category_3','product_type'])

dfDesc=af.getDFDesc(item_data)
item_data['item_price_log']=np.log(item_data['item_price'])

dfDesc=af.getDFDesc(log_data)

log_data=af.categorizeCols(log_data,cols=['session_id','user_id','item_id'])

dfDesc=af.getDFDesc(log_data)

#log_data['server_time'] = log_data['server_time'].apply(dateutil.parser.parse)
#train_test_data['impression_time'] = train_test_data['impression_time'].apply(dateutil.parser.parse)


"""
log_data_copy=log_data.copy()

log_data_copy=log_data.copy()
log_data_user_session = log_data.set_index(['user_id','session_id'])
log_data_user_session=log_data_user_session.sort_index(level=[0,1])

u48=log_data_user_session[log_data_user_session.index.get_level_values('user_id').isin(['79202'])]

train_test_data_user=train_test_data.set_index(['user_id','is_click'])
train_test_data_user=train_test_data_user.sort_index(level=[0,1])

train_test_data_1=train_test_data[train_test_data.is_click==1]

uid='52530'
u48_im=train_test_data_user[train_test_data_user.index.get_level_values('user_id').isin([uid])]
u48=log_data_user_session[log_data_user_session.index.get_level_values('user_id').isin([uid])]


log_data_click_1=log_data[log_data]


train_log=pd.merge(train_test_data,log_data,left_on=['user_id','impression_time'],right_on=['user_id','server_time'])


train_test_data_1_user=train_test_data_1[train_test_data_1['user_id'].isin(log_data['user_id'])==False]
"""

log_item_data=pd.merge(log_data,item_data,on='item_id')

train_test_data['impression_time_7less'] = train_test_data['impression_time'] -  pd.to_timedelta(7, unit='d')
featurelist=['diff1','diff2','diff3',  'item_price_avg',  'item_price_max',  'item_price_min',  'item_price_std',    'previous_imp_app_count',  'previous_imp_count',  'total_category_1',  'total_category_2',  'total_category_3',  'total_items',  'total_product_type',  'total_sessions',  'visit_count']
for column in featurelist:
	train_test_data[column]=-1


train_test_data.reset_index(inplace=True,drop=True)



log_data_user = log_item_data.set_index(['user_id'])
train_test_data_user = train_test_data.set_index(['user_id'])


user_ids_list =  np.unique(log_data['user_id'])
#for n,user_id in enumerate(user_ids_list):
def generateFeatures(user_id):
	f1=train_test_data_user.index.get_level_values('user_id').isin([user_id])
	f1_log=log_data_user.index.get_level_values('user_id').isin([user_id])
	
	mydf=train_test_data[f1]
	
	log_item_data_u=log_item_data[f1_log]
	if user_id%100==0:
		print(user_id,len(mydf))

	for i,row in mydf.iterrows():
		
		app_id=row['app_code']
		it=row['impression_time']
		it_seven_less=row['impression_time_7less']
		
		
		log_item_data_ut=log_item_data_u[log_item_data_u['server_time'] < it_seven_less]
		#log_item_data_ut=log_item_data_ut[f2_log]
		#log_item_data_ut=log_data_user_session[log_data_user_session.index.get_level_values('server_time')<it_seven_less]
		#log_item_data_ut=log_item_data[(log_item_data['user_id']==user_id) & (log_item_data['server_time']<it_seven_less)]
		"""train_test_data.loc[i,'visit_count']=len(log_item_data_ut)
		if len(log_item_data_ut)>0:
			train_test_data.loc[i,'total_sessions']=len(np.unique(log_item_data_ut['session_id']))
			train_test_data.loc[i,'total_items']=len(np.unique(log_item_data_ut['item_id']))
			train_test_data.loc[i,'total_category_1']=len(np.unique(log_item_data_ut['category_1']))
			train_test_data.loc[i,'total_category_2']=len(np.unique(log_item_data_ut['category_2']))
			train_test_data.loc[i,'total_category_3']=len(np.unique(log_item_data_ut['category_3']))
			train_test_data.loc[i,'total_product_type']=len(np.unique(log_item_data_ut['product_type']))
			
			train_test_data.loc[i,'item_price_max']=np.max(log_item_data_ut['item_price_log'])
			train_test_data.loc[i,'item_price_min']=np.min(log_item_data_ut['item_price_log'])
			train_test_data.loc[i,'item_price_avg']=np.mean(log_item_data_ut['item_price_log'])
			train_test_data.loc[i,'item_price_std']=np.std(log_item_data_ut['item_price_log'])
			max_time=np.max(log_item_data_ut['server_time'])
			diff=it - max_time
			train_test_data.loc[i,'diff1']=diff.total_seconds()
		"""
		
		imp_data=mydf[mydf['impression_time'] < it_seven_less]
		"""train_test_data.loc[i,'previous_imp_count']=len(imp_data)
		if len(imp_data)>0:
			max_time=np.max(imp_data['impression_time'])
			diff=it - max_time
			train_test_data.loc[i,'diff2']=diff.total_seconds()
			train_test_data.loc[i,'previous_imp_app_count']=len(np.unique(imp_data['app_code']))
			
			
			imp_data_app=imp_data[imp_data['app_code']==app_id]
			train_test_data.loc[i,'previous_imp_app_count']=len(imp_data_app)
			if len(imp_data_app)>0:
				max_time=np.max(imp_data_app['impression_time'])
				diff=it - max_time
				train_test_data.loc[i,'diff3']=diff.total_seconds()
				
			#if n%100==0:
				#print(n,len(mydf))
		"""


for user_id in user_ids_list:
	generateFeatures(user_id)	 


train_test_data['it_day_of_week'] = train_test_data['impression_time'].dt.dayofweek
train_test_data['it_month_start'] = train_test_data['impression_time'].dt.is_month_start
train_test_data['it_month_end'] = train_test_data['impression_time'].dt.is_month_end
train_test_data['it_weekday'] = train_test_data['impression_time'].apply(lambda x: x.weekday())

# because of high cardinallity code app_code as overall number of visits
s=train_test_data.app_code.value_counts()
s=s/len(train_test_data)
train_test_data['app_id']=train_test_data['app_code'].apply(lambda x: s[x])


train_test_data['diff_days']=math.floor(train_test_data['diff1']/3600/24)
train_test_data['diff_hours']=math.floor(train_test_data['diff1']/3600)
train_test_data['diff_mins']=math.floor(train_test_data['diff1']/60)
train_test_data['diff_secs']=math.floor(train_test_data['diff1'])

train_test_data['prev_diff_days']=math.floor(train_test_data['diff2']/3600/24)
train_test_data['prev_diff_hours']=math.floor(train_test_data['diff2']/3600)
train_test_data['prev_diff_mins']=math.floor(train_test_data['diff2']/60)
train_test_data['prev_diff_secs']=math.floor(train_test_data['diff2'])
train_test_data['prev_app_diff_days']=math.floor(train_test_data['diff3']/3600/24)
train_test_data['prev_app_diff_hours']=math.floor(train_test_data['diff3']/3600)
train_test_data['prev_app_diff_mins']=math.floor(train_test_data['diff3']/60)
train_test_data['prev_app_diff_secs']=math.floor(train_test_data['diff3'])


			
		
"""import dask.dataframe as dd
from dask.multiprocessing import get
ddata = dd.from_pandas(train_test_data, npartitions=10)
ddata.map_partitions(lambda df: df.apply((lambda row: generateFeatures(*row)), axis=1)).compute(get=get)
"""

	
train_test_data=train_test_data.drop(columns=['impression_id','impression_time','user_id','impression_time_7less','app_code','diff1','diff2','diff3'])	
train_test_data=af.categorizeCols(train_test_data,cols=['os_version','it_day_of_week','it_weekday'])
train_test_data=af.LabelEncodeCols(train_test_data.copy(),categorical_columns=[],onehotColumns=['os_version','it_day_of_week','it_weekday'])	


	
X=train_test_data    

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


res,PredDF,predtrain=hh.GetBestH2OModel(trainVal_frame,x_cols,y_col,pred_variable_type == "categorical",X_test)
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
	
	






















