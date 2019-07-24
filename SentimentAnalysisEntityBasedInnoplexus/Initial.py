"""

Problem Statement: 
    1. Sentiment Analysis for given Drug Name
    2. Same statement, different sentiment for a drug 
    3. Coreference resolution

"""


import pandas as pd
import numpy as np
import AllFunctions as af
import re

dftr=pd.read_csv('train.csv')
dfts=pd.read_csv('test.csv')

df=pd.concat([dftr,dfts],ignore_index=True)

dfDescT = af.getDFDesc(df)

dfhead=df.head(5)

"""
Observations:
    
    1. There are 111 different types of drugs in train and test dataset
    2. Few text give reviews for multiple drugs(upto 6 in given dataset)
    3. Many medical jargons/Abbr. included in text.
    
"""

"""
EDA:
    1. wordcount
    2. word and sentiment combo
"""

allwords=[]


eda_res=[]
drug_list=list(set(df['drug']))
"""
drug_list_sent=[drug+"_"+str(sentiment) for drug in drug_list for sentiment in range(3)]
drug_word_df=pd.DataFrame(columns=drug_list_sent)
drug_word_df=drug_word_df.T
"""

googlevector='C:\\Work\\Downloads\\Libs\\GoogleNews-vectors-negative300.bin'
import gensim

# Load Google's pre-trained Word2Vec model.
model = gensim.models.KeyedVectors.load_word2vec_format(googlevector, binary=True) 

df['CleanText']=df['text']

dfhead=df.head(5)

dfvector=pd.DataFrame(index=range(len(df)),columns=range(300))
def avg_sentence(sentence, wv):
  v = np.zeros(300)
  for w in sentence:
    if w in wv:
      v += wv[w]
  return v / len(sentence)


for index,row in df.iterrows():
    print(index)
    text=row['text']
    drug=row['drug']
    sentiment=row['sentiment']
    
    text=re.sub(r'[^A-Za-z\s]','',text) # nums, punc
    text=re.sub(r'[\s]+',' ',text).lower() #multiple spaces and case
    
    df.loc[index,'CleanText']=text
    words=text.split(' ')
    
    allwords.append(words)
    vectors = avg_sentence(words, model)
    dfvector.loc[index]=vectors  
    
dfvector.to_csv('dfvector.csv',index=False) 
 
dfvector.columns=['TV'+str(i) for i in range(len(dfvector.columns))]
       
drugvector=[]    
for drug in drug_list: 
    if drug in model:
        print(drug)
        drugvector.append(drug)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(df['drug'])
encoded_Y = encoder.transform(df['drug'])    
from keras.utils import to_categorical    
dfdrug=to_categorical(encoded_Y)     

dfdrug=pd.DataFrame(dfdrug)
dfdrug.columns=['D'+str(i) for i in range(len(dfdrug.columns))]

dfnew=pd.concat([dfvector,dfdrug],axis=1)

train_val=dfnew.loc[0:len(dftr)-1] 
test=dfnew.loc[len(dftr):]  

train_val_y=to_categorical(dftr['sentiment'])


from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

from keras import backend as K

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# define baseline model
def baseline_model(input_dim,output_dim):
	# create model
	model = Sequential()
	model.add(Dense(32, input_dim=input_dim, activation='relu'))
	model.add(Dropout(0.1))
	model.add(Dense(8,  activation='relu'))
	model.add(Dropout(0.1))
	model.add(Dense(output_dim, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc',f1_m])
	return model  

    
estimator = KerasClassifier(build_fn=baseline_model,input_dim=len(train_val.columns),output_dim=train_val_y.shape[1], epochs=50, batch_size=50, verbose=True)   
estimator.fit(train_val, train_val_y)

kfold = KFold(n_splits=5, shuffle=True, random_state=7) 
results = cross_val_score(estimator, train_val, train_val_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
    
    
test_res=estimator.predict(test)   
dfts['sentiment']=test_res 
sample=dfts[['unique_hash','sentiment']]   
sample.to_csv('sample_submission1.csv',index=False)    
    
    
    
    
    