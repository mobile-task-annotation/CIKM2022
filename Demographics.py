
# coding: utf-8

import numpy as np
import pandas as pd
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import nltk
from gensim.models import Word2Vec




from scipy.stats import zscore

import matplotlib.pyplot as plt
import random 

from scipy import stats
from sklearn.model_selection import KFold




from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict


from sklearn import preprocessing
from xgboost import XGBClassifier


from collections import Counter
import seaborn as sns
from ast import literal_eval
import imblearn
from imblearn.over_sampling import RandomOverSampler # doctest: +NORMALIZE_WHITESPACE
import collections





#### Read the dataset
#Logs: Users' app usage logs with user_id, app_id and task_id
#User_info: Users' info with gender and age




###Baseline Doc2Vec
Logs['app_id'] = Logs['app_id'].astype(str)
User_App_Usage = Logs.groupby('user_id')['app_id'].apply(list)
tagged_data = [TaggedDocument(words = _d, tags = [str(i)]) for i,_d in enumerate(Logs.groupby('user_id')['app_id'].apply(list))]
max_epochs = 100
vec_size = 500
alpha = 0.025

model = Doc2Vec(vector_size=vec_size,
                alpha=alpha,
                min_alpha=0.00025,
                min_count=1,
                   window = 4,
                dm=1)
model.build_vocab(tagged_data)







for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha





Log_Users = Logs['user_id'].unique()





all_user_array=[]

for user in Log_Users:
  apps = User_App_Usage[user]
  user_app_array = [user]
  embeddings = np.zeros(vec_size)
  for app in apps:
    embeddings = np.add(embeddings, model.infer_vector(apps))
  user_app_array.extend(embeddings)
  all_user_array.append(user_app_array)



Logs_with_Task_ID_info = Logs_with_Task_ID[['user_id', 'task_id', 'app_id', 'Label_complete47']].copy()
#User_info[User_info['user_id'].isin(Logs_with_Task_ID_info['user_id'].unique())].groupby('gender').size()
#User_info[User_info['user_id'].isin(Logs_with_Task_ID_info['user_id'].unique())].groupby('age_range_min').size()






all_apps = Logs['mobile_app_category'].unique()



gender_female = collections.defaultdict(list)
gender_male = collections.defaultdict(list)
age_13 = collections.defaultdict(list)
age_18 = collections.defaultdict(list)
age_25 = collections.defaultdict(list)
age_35 = collections.defaultdict(list)
age_55 = collections.defaultdict(list)

for app in all_apps:
  selected_df = Logs[Logs['mobile_app_category'] == app].copy()
  app_user = selected_df['user_id'].unique()
  app_user = User_info[User_info['user_id'].isin(app_user)]
  age_df = pd.DataFrame(app_user.groupby('age_range_min').size()/app_user.shape[0]).reset_index()

  for i in range(age_df.shape[0]):
    cur_age = age_df.loc[i, 'age_range_min']

    if cur_age == 13:
      age_13[app].append(age_df.iloc[i,1])
    if cur_age == 18:
      age_18[app].append(age_df.iloc[i,1])
    if cur_age == 25:
      age_25[app].append(age_df.iloc[i,1])
    if cur_age == 35:
      age_35[app].append(age_df.iloc[i,1])
    if cur_age == 55:
      age_55[app].append(age_df.iloc[i,1])
   





sorted(age_55.items(), key = lambda x: x[1], reverse = True)





sorted(gender_male.items(), key = lambda x: x[1], reverse = True)





sorted(gender_female.items(), key = lambda x: x[1], reverse = True)





'''
Task_ID_Users = Logs_with_Task_ID['user_id'].unique()
Logs_Filter = Logs[Logs['user_id'].isin(Task_ID_Users)].reset_index(drop = True).copy()

User_Tasks_App_Amount = pd.DataFrame({'app_amount':Logs.groupby(['user_id', 'task_id'])['app_id'].nunique()}).reset_index()
User_Tasks_App_Amount[User_Tasks_App_Amount['app_amount']>=2].shape
'''





#Logs_with_Task_ID_info['Label_complete47'] = Logs_with_Task_ID_info['Label_complete47'].astype(str)
#Logs_with_Task_ID_info['app_id'] =Logs_with_Task_ID_info['app_id'].astype(str)
Logs['app_id'] = Logs['app_id'].astype(str)





'''
User_App_Amount = pd.DataFrame({'app_amount':Logs.groupby('user_id')['app_id'].nunique()}).reset_index()

User_App_Amount_filter = User_App_Amount[User_App_Amount['app_amount']<=15]
User_App_Amount_filter
Logs = Logs[Logs['user_id'].isin(User_App_Amount_filter['user_id'])].reset_index(drop = True).copy()'''




#embedding based on tasks instead of specific apps
# assign each task the specific app sequence string

Tasks = pd.DataFrame({'Task_apps':Logs.groupby(['user_id', 'task_id'])['app_id'].apply(lambda x: ' '.join(x))}).reset_index()

# assign each task the unique app string
#Tasks= pd.DataFrame({'Task_apps':Logs_with_Task_ID_info.groupby(['user_id', 'task_id'])['Label_complete47'].unique().apply(lambda x: ' '.join(x))}).reset_index()



# assign each task the unique  Task ID
#Tasks = pd.DataFrame({'Task_apps':Logs_with_Task_ID_info.groupby(['user_id', 'task_id'])['app_id'].unique().apply(lambda x: ','.join(x))}).reset_index()








# original word2vec embeddings based on specific apps 
Global_list = Logs.groupby(['user_id'])['app_id'].apply(list).tolist()


### word2vec embeddings based on tasks
#Global_list = Tasks.groupby(['user_id'])['Task_apps'].apply(list).tolist()

### embedding based on apps and task id
#Global_list = Logs_with_Task_ID.groupby(['user_id'])['task_type'].apply(list).tolist()





sentences = Global_list
embedding_size = 300
window_size = 8

model_wv = Word2Vec(sentences, min_count=1, size = embedding_size, window = window_size)





Users_App_Unique = Logs.groupby('user_id')['app_id'].unique()

#Users_App_Unique = Tasks.groupby('user_id')['Task_apps'].unique()

#Users_App_Unique = Logs_with_Task_ID.groupby('user_id')['task_type'].unique()
Log_Users = Users_App_Unique.index





all_user_array=[]

for user in Log_Users:
  apps = Users_App_Unique[user]
  user_app_array = [user]
  embeddings = np.zeros(embedding_size)
  for app in apps:
    embeddings = np.add(embeddings, model_wv.wv[app])
  user_app_array.extend(embeddings)
  all_user_array.append(user_app_array)






user_app_embeddings = pd.DataFrame(all_user_array)
user_app_embeddings = user_app_embeddings.fillna(0)

user_app_embeddings = user_app_embeddings.rename(columns={0: 'user_id'})





#user_app_embeddings = pd.merge(user_app_embeddings, user_task_vector, on = 'user_id', how = 'left').fillna(0)






#user_app_embeddings = User_app_id.copy()


user_app_embeddings_demo = pd.merge(user_app_embeddings, User_info[['user_id', 'age_range_min', 'gender']], how = 'inner', on = 'user_id')
#user_app_embeddings_demo = user_app_embeddings_demo[user_app_embeddings_demo['user_id'].isin(user_filter)]

#user_filter = user_app_embeddings_demo.user_id

user_app_embeddings_demo 





task = 'gender'

def split_df(dataframe, column_name, training_split, validation_split, test_split):
    """
    Splits a pandas dataframe into trainingset, validationset and testset in specified ratio.
    All sets are balanced, which means they have the same ratio for each category as the full set.
    Input:   dataframe        - Pandas Dataframe, should include a column for data and one for categories
             column_name      - Name of dataframe column which contains the categorical output values
             training_split   - from ]0,1[, default = 0.6
             validation_split - from ]0,1[, default = 0.2        
             test_split       - from ]0,1[, default = 0.2
                                Sum of all splits need to be 1
    Output:  train            - Pandas DataFrame of trainset
             validation       - Pandas DataFrame of validationset
             test             - Pandas DataFrame of testset
    """
    if training_split + validation_split + test_split != 1.0:
        raise ValueError('Split paramter sum should be 1.0')
        
    total = len(dataframe.index)
 
    train = dataframe.reset_index().groupby(column_name).apply(lambda x: x.sample(frac=training_split, random_state = 10))    .reset_index(drop=True).set_index('index')
    train = train.sample(frac=1)
    temp_df = dataframe.drop(train.index)
    validation = temp_df.reset_index().groupby(column_name)    .apply(lambda x: x.sample(frac=validation_split/(test_split+validation_split), random_state = 5))           .reset_index(drop=True).set_index('index')
    validation = validation.sample(frac=1, random_state = 5)
    test = temp_df.drop(validation.index)
    test = test.sample(frac=1, random_state = 5)
    
    print('Total: ', len(dataframe))
    print('Training: ', len(train), ', Percentage: ', len(train)/len(dataframe))
    print('Validation: ', len(validation), ', Percentage: ', len(validation)/len(dataframe))
    print('Test:', len(test), ', Percentage: ', len(test)/len(dataframe))

    return train, validation, test

Train, Validation, Test = split_df(user_app_embeddings_demo, task,0.8,0.1,0.1)





id_f1_list = []


Classifier_List = ['RF']
#'LR', 'KNN', 'RF', 

    
for i in range(1):
  for c in Classifier_List:
    if c == 'LR':
      classifier = LogisticRegression(penalty = 'l2', C = .01)
    elif c == 'RF':
      classifier = RandomForestClassifier()
    elif c == 'SVM':
      classifier = SVC()
    elif c == 'KNN':
      classifier = KNeighborsClassifier(n_neighbors=3)
    elif c == 'XGB':
      classifier = XGBClassifier()
        
    print('%s............................................'%(c))
 
  
    print(Train.shape)
    print(Test.shape)

    clf = classifier.fit(Train.iloc[:, 1:-2], Train[task])
    result = clf.predict(Test.iloc[:,1:-2]).copy()
    



    print("weighted ##### acc:%f, pre:%f, rec:%f, f1:%f"%(accuracy_score(result,Test[task]), 
                                                      precision_score(result,Test[task], average='weighted'),
                                                      recall_score(result,Test[task], average='weighted'),
                                                      f1_score(result,Test[task], average='weighted')
                                                      ))
    print("macro ##### acc:%f, pre:%f, rec:%f, f1:%f"%(accuracy_score(result,Test[task]), 
                                                      precision_score(result,Test[task], average='macro'),
                                                      recall_score(result,Test[task], average='macro'),
                                                      f1_score(result,Test[task], average='macro')
                                                      ))

    id_f1_list.append(f1_score(result,Test[task], average='macro'))





All_gender_Results = pd.DataFrame(Test[['user_id', 'gender']].copy()).reset_index(drop = True)
All_gender_Results['RF_app2vec_gender'] = result
#All_age_Results = pd.DataFrame(Test[['user_id', 'age_range_min']].copy()).reset_index(drop = True)
#All_age_Results['LR_app2vec_age'] = result


# In[1]:


#### Generate the app id features of each user
User_app_id = pd.get_dummies(Users_App_Unique.apply(pd.Series).stack()).sum(level=0).reset_index()





### Bag-of-Tasks-Types

Users_Task_App_Unique = pd.DataFrame({'Task_Unique_App':Logs.groupby(['user_id', 'task_id'])['app_id'].unique()}).reset_index()

Users_Task = pd.DataFrame({'task_list':Users_Task_App_Unique.groupby('user_id')['Task_Unique_App'].apply(list)}).reset_index()
Users_Task['task_set'] = Users_Task['task_list'].apply(lambda x: [list(y) for y in set(tuple(y) for y in x)])
Users_Task['task_set'] = Users_Task['task_set'].apply(lambda x: [str(i) for i in x])
User_app_id = pd.concat([Users_Task['user_id'], pd.get_dummies(Users_Task.task_set.apply(pd.Series).stack()).sum(level=0).reset_index(drop = True)], axis = 1)
User_app_id





def task_type_assign(row):
  if row['app_id'] == row['Label_complete47']:
    return row['mobile_app_category']
  else:
    return row['Label_complete47']

Logs_with_Task_ID['task_type'] = Logs_with_Task_ID.apply(task_type_assign, axis = 1)





Logs_with_Task_ID['task_type']  = Logs_with_Task_ID['task_type'].astype(str)





Users_App_Unique = Logs_with_Task_ID.groupby('user_id')['Label_complete47'].unique()
User_app_id = pd.get_dummies(Users_App_Unique.apply(pd.Series).stack()).sum(level=0).reset_index()





### Task2Vec: Concatenate app embeddings and task embeddings


context_vector_demographics['sum_context_vector'] = context_vector_demographics['sum_context_vector'].apply(lambda x: x[0:1]+' '+x[1:-1] + ' ' + x[-1:])
task_vector = pd.DataFrame(context_vector_demographics['sum_context_vector'].str.split().values.tolist())
user_task_vector = pd.concat([context_vector_demographics['user_id'], task_vector.iloc[:, 1:-1]], axis = 1)





#### CNN
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
#from keras.utils import to_categorical
from keras.layers import Embedding


from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model,Sequential

from keras.layers import Input, Dense, Embedding, Conv2D, MaxPooling2D, Dropout,concatenate, SpatialDropout1D, LSTM
from keras.layers.core import Reshape, Flatten
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.models import Model
from keras import regularizers


import string

from keras.layers import Bidirectional, GRU, TimeDistributed, Layer, Activation, Dot

from keras import backend as K, initializers, regularizers, constraints






# assign each task the unique  Task ID
Task_texts = pd.DataFrame({'app_texts':Tasks.groupby(['user_id'])['Task_apps'].apply(list)}).reset_index()
#Task_texts = pd.DataFrame({'app_texts': Logs_with_Task_ID.groupby(['user_id'])['task_type'].unique().apply(list)}).reset_index()

#Tasks.groupby(['user_id'])['Task_apps'].apply(list).tolist()



### embedding based on apps
#Task_texts = pd.DataFrame({'app_texts':Logs.groupby(['user_id'])['app_id'].apply(list)}).reset_index()




texts_demo = pd.merge(Task_texts, User_info[['user_id', 'age_range_min', 'gender']], on = 'user_id', how = 'inner')






task = 'age_range_min'

train_data, val_data, test_data = split_df(texts_demo, task, 0.8,0.1,0.1)





'''
task = 'age_range_min'

ground_truth =texts_demo.age_range_min.unique()
dic={}
for i, age in enumerate(ground_truth):
    dic[age]=i
labels=texts_demo.age_range_min.apply(lambda x:dic[x])

'''

ground_truth =texts_demo.gender.unique()
dic={}
for i, gender in enumerate(ground_truth):
    dic[gender]=i
labels=texts_demo.gender.apply(lambda x:dic[x])





texts = texts_demo.app_texts

tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'')
tokenizer.fit_on_texts(texts)
sequences_train = tokenizer.texts_to_sequences(train_data.app_texts)
sequences_valid=tokenizer.texts_to_sequences(val_data.app_texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))





#np.quantile(texts.apply(lambda x: len(x)), 0.9)





#MAXSEQLENGTH = max(texts.apply(lambda x: len(x)))
MAXSEQLENGTH = np.quantile(texts.apply(lambda x: len(x)), 0.9)
MAXSEQLENGTH = 9





X_train = pad_sequences(sequences_train, maxlen = MAXSEQLENGTH)
X_val = pad_sequences(sequences_valid, maxlen=MAXSEQLENGTH)
y_train = to_categorical(np.asarray(labels[train_data.index]))
y_val = to_categorical(np.asarray(labels[val_data.index]))
print('Shape of X train and X validation tensor:', X_train.shape,X_val.shape)
print('Shape of label train and validation tensor:', y_train.shape,y_val.shape)





#### with pre-trained word2vec embeddings

EMBEDDING_DIM=300
vocabulary_size = len(word_index)+1
embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))

for word, i in word_index.items():
    try:
        embedding_vector = model_wv.wv[word]
        embedding_matrix[i] = embedding_vector
    except KeyError:
        embedding_matrix[i]=np.random.normal(0,np.sqrt(0.25),EMBEDDING_DIM)
#del(word_vectors)


embedding_layer = Embedding(vocabulary_size,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            trainable=True)





#### without pre-trained word2vec embeddings
EMBEDDING_DIM=600
vocabulary_size= len(word_index)+1

embedding_layer = Embedding(vocabulary_size,
                            EMBEDDING_DIM)





sequence_length = MAXSEQLENGTH
category_size = 5
#category_size = 2

filter_sizes = [3,4,5,6,7]


num_filters = 100
drop = 0.5



inputs = Input(shape=(sequence_length,))
embedding = embedding_layer(inputs)
reshape = Reshape((sequence_length,EMBEDDING_DIM,1))(embedding)

conv_0 = Conv2D(num_filters, (filter_sizes[0], EMBEDDING_DIM),activation='relu',kernel_regularizer=regularizers.l2(0.01))(reshape)
conv_1 = Conv2D(num_filters, (filter_sizes[1], EMBEDDING_DIM),activation='relu',kernel_regularizer=regularizers.l2(0.01))(reshape)
conv_2 = Conv2D(num_filters, (filter_sizes[2], EMBEDDING_DIM),activation='relu',kernel_regularizer=regularizers.l2(0.01))(reshape)
conv_3 = Conv2D(num_filters, (filter_sizes[3], EMBEDDING_DIM),activation='relu',kernel_regularizer=regularizers.l2(0.01))(reshape)
conv_4 = Conv2D(num_filters, (filter_sizes[4], EMBEDDING_DIM),activation='relu',kernel_regularizer=regularizers.l2(0.01))(reshape)



maxpool_0 = MaxPooling2D((sequence_length - filter_sizes[0] + 1, 1), strides=(1,1))(conv_0)
maxpool_1 = MaxPooling2D((sequence_length - filter_sizes[1] + 1, 1), strides=(1,1))(conv_1)
maxpool_2 = MaxPooling2D((sequence_length - filter_sizes[2] + 1, 1), strides=(1,1))(conv_2)
maxpool_3 = MaxPooling2D((sequence_length - filter_sizes[3] + 1, 1), strides=(1,1))(conv_3)
maxpool_4 = MaxPooling2D((sequence_length - filter_sizes[4] + 1, 1), strides=(1,1))(conv_4)




merged_tensor = concatenate([maxpool_0, maxpool_1, maxpool_2, maxpool_3, maxpool_4], axis=1)
#merged_tensor = concatenate([maxpool_0, maxpool_1], axis=1)

flatten = Flatten()(merged_tensor)
reshape = Reshape((category_size*num_filters,))(flatten)
dropout = Dropout(drop)(flatten)
output = Dense(units=category_size, activation='softmax',kernel_regularizer=regularizers.l2(0.01))(dropout)

# this creates a model that includes
model = Model(inputs,  output)

model.summary()





adam = Adam(lr=1e-3)

model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

callbacks = [EarlyStopping(monitor='val_loss', patience = 2)]



#class_weight = class_weights, 

model.fit(X_train, y_train, batch_size=128, epochs=50,  validation_data=(X_val, y_val),
         callbacks=callbacks)  # starts training





sequences_test=tokenizer.texts_to_sequences(test_data.app_texts)
X_test = pad_sequences(sequences_test,maxlen=MAXSEQLENGTH)
y_pred=model.predict(X_test)





to_submit=pd.DataFrame(index=test_data.user_id,data={13:y_pred[:,dic[13]],
                                                18:y_pred[:,dic[18]],
                                                25:y_pred[:,dic[25]],
                                                35:y_pred[:,dic[35]],
                                                55:y_pred[:,dic[55]]}).reset_index()





to_submit=pd.DataFrame(index=test_data.user_id,data={'f':y_pred[:,dic['f']],
                                                'm':y_pred[:,dic['m']]}).reset_index()





to_submit['predicted'] = to_submit.iloc[:, 1:].idxmax(axis=1)
to_submit_demo = pd.merge(to_submit, User_info[['user_id', 'gender', 'age_range_min']], on = 'user_id', how = 'inner')


task = 'age_range_min'
print("weighted ##### acc:%f, pre:%f, rec:%f, f1:%f"%(accuracy_score(to_submit_demo['predicted'], to_submit_demo[task]), 
                                                      precision_score(to_submit_demo['predicted'], to_submit_demo[task], average='weighted'),
                                                      recall_score(to_submit_demo['predicted'], to_submit_demo[task], average='weighted'),
                                                      f1_score(to_submit_demo['predicted'], to_submit_demo[task], average='weighted')
                                                      ))
print("macro ##### acc:%f, pre:%f, rec:%f, f1:%f"%(accuracy_score(to_submit_demo['predicted'], to_submit_demo[task]), 
                                                      precision_score(to_submit_demo['predicted'], to_submit_demo[task], average='macro'),
                                                      recall_score(to_submit_demo['predicted'], to_submit_demo[task], average='macro'),
                                                      f1_score(to_submit_demo['predicted'], to_submit_demo[task], average='macro')
                                                      ))







######################HURA

#!/usr/bin/env python
# coding: utf-8

# In[1]:

import re
import codecs
import numpy as np
from sklearn.metrics import classification_report





### Prepare the data

Tasks = pd.DataFrame({'Task_apps':Logs.groupby(['user_id', 'task_id'])['app_id'].unique().apply(lambda x: ' '.join(x))}).reset_index()


Task_texts = pd.DataFrame({'app_texts':Tasks.groupby(['user_id'])['Task_apps'].apply(lambda x: '#'.join(x))}).reset_index()
texts_demo = pd.merge(Task_texts, User_info[['user_id', 'age_range_min', 'gender']], on = 'user_id', how = 'inner')





texts_demo['gender'] = texts_demo['gender'].apply(lambda x: 0 if x == 'f' else 1)

def age_transfer(age_group):
  if age_group == 13:
    return 0
  elif age_group == 18:
    return 1
  elif age_group == 25:
    return 2
  elif age_group == 35:
    return 3
  else:
    return 4
    
texts_demo['age_range_min'] = texts_demo['age_range_min'].apply(age_transfer)





'''

file = codecs.open('/content/drive/My Drive/google_colab/processing_data/sample_userdata.txt', "r")

user_query = []
user_label = []



for line in file.readlines():
    print(line)
    terms = line.split('\t')
    text = terms[1].lower()
    sentences = text.split('#')

    user_query.append([x.split() for x in sentences])
    user_label.append(int(terms[0]))
file.close()
'''





#### Parameters setting for HURA


task = 'gender'
maxlen= 2 ## 95% are less than 2, 80% are less than 1
maxsent = 70  ## 90% are less than 118,  80% are less than 71
task_nodes_num = 2  ## number of task nodes: age 5 gender 2
EMBEDDING_DIM=300







user_query = []
user_label = []

for i in range(len(texts_demo.index)):
    sentences = texts_demo.loc[i, 'app_texts'].split('#')
    user_query.append([x.split() for x in sentences])
    user_label.append(int(texts_demo.loc[i, task]))






# In[2]:
####build the word_dict:   [0]: index  [1]: frequency  

#word_dict={'PADDING':[0,999999],'UNK':[1,99999]}

word_dict = {}

for i in user_query:
    for sent in i:
        for word in sent:
            if not word in word_dict:
                word_dict[str(word)]=[len(word_dict),1]
            else:
                word_dict[str(word)][1]+=1










embdict=dict()

for word in word_dict:
  try:
    embedding_vector = model_wv.wv[word]
    embdict[word] = embedding_vector
  except KeyError:
    print("KeyError")
    embdict[word] = np.random.normal(0,np.sqrt(0.25),EMBEDDING_DIM)







# In[6]:


from numpy.linalg import cholesky

lister=[0]*len(word_dict)
xp=np.zeros(EMBEDDING_DIM,dtype='float32')

cand=[]

for i in embdict.keys():
    lister[word_dict[i][0]]=np.array(embdict[i],dtype='float32')
    cand.append(lister[word_dict[i][0]])
cand=np.array(cand,dtype='float32')


mu=np.mean(cand, axis=0)
Sigma=np.cov(cand.T)


norm=np.random.multivariate_normal(mu, Sigma, 1)
print(mu.shape,Sigma.shape,norm.shape)



for i in range(len(lister)):
    if type(lister[i])==int:
        lister[i]=np.reshape(norm, EMBEDDING_DIM)



lister[0]=np.zeros(EMBEDDING_DIM, dtype='float32')
lister=np.array(lister, dtype='float32')
print(lister.shape)





np.quantile(Logs.groupby(['user_id'])['task_id'].size(),0.95)






# In[7]:

'''

user_query_data=[]
for i in user_query:
    userdata=[]
    for sent in i:
        sentence=[]
        for word in sent:
            if word in word_dict:
                 sentence.append(word_dict[str(word)][0])
            if len(sentence)==maxlen:
                break
        userdata.append(sentence+[0]*(maxlen-len(sentence)))
        if len(userdata)==maxsent:
            break
    user_query_data.append(userdata+[[0]*maxlen]*(maxsent-len(userdata)))
'''





def split_df(dataframe, column_name, training_split, validation_split, test_split):
    """
    Splits a pandas dataframe into trainingset, validationset and testset in specified ratio.
    All sets are balanced, which means they have the same ratio for each category as the full set.
    Input:   dataframe        - Pandas Dataframe, should include a column for data and one for categories
             column_name      - Name of dataframe column which contains the categorical output values
             training_split   - from ]0,1[, default = 0.6
             validation_split - from ]0,1[, default = 0.2        
             test_split       - from ]0,1[, default = 0.2
                                Sum of all splits need to be 1
    Output:  train            - Pandas DataFrame of trainset
             validation       - Pandas DataFrame of validationset
             test             - Pandas DataFrame of testset
    """
    if training_split + validation_split + test_split != 1.0:
        raise ValueError('Split paramter sum should be 1.0')
        
    total = len(dataframe.index)
 
    train = dataframe.reset_index().groupby(column_name).apply(lambda x: x.sample(frac=training_split, random_state = 10))    .reset_index(drop=True).set_index('index')
    train = train.sample(frac=1)
    temp_df = dataframe.drop(train.index)
    validation = temp_df.reset_index().groupby(column_name)    .apply(lambda x: x.sample(frac=validation_split/(test_split+validation_split), random_state = 5))           .reset_index(drop=True).set_index('index')
    validation = validation.sample(frac=1, random_state = 5)
    test = temp_df.drop(validation.index)
    test = test.sample(frac=1, random_state = 5)
    
    print('Total: ', len(dataframe))
    print('Training: ', len(train), ', Percentage: ', len(train)/len(dataframe))
    print('Validation: ', len(validation), ', Percentage: ', len(validation)/len(dataframe))
    print('Test:', len(test), ', Percentage: ', len(test)/len(dataframe))

    return train, validation, test

train, validation, test = split_df(texts_demo, task,0.8,0.1,0.1)





test = texts_demo[texts_demo.user_id.isin(Test.user_id.unique())].reset_index(drop = True)
train = texts_demo[texts_demo.user_id.isin(Train.user_id.unique())].reset_index(drop = True)
validation = texts_demo[texts_demo.user_id.isin(Validation.user_id.unique())].reset_index(drop = True)





'''
# define oversampling strategy
oversample = RandomOverSampler(sampling_strategy='minority')
# fit and apply the transform
X_over, y_over = oversample.fit_resample(X_over, y_over)


print(Counter(y_over))
train_over = pd.DataFrame(X_over)
train_over.columns = train.columns'''









df = ['train', 'test', 'val']

for d in df:
  if d == 'train':
    df = train.reset_index(drop = True)
  elif d == 'test':
    df = test.reset_index(drop = True)
  elif d == 'val':
    df = validation.reset_index(drop = True)

  print(df.shape)

  user_query = []
  user_label = []

  for i in range(len(df.index)):
    sentences = df.loc[i, 'app_texts'].split('#')
    user_query.append([x.split() for x in sentences])
    user_label.append(int(df.loc[i, task]))



  user_query_data= []


  for i in user_query:
    userdata=[]
    for sent in i:
        sentence=[]
        for word in sent:
            if word in word_dict:
                 sentence.append(word_dict[str(word)][0])
            if len(sentence)==maxlen:
                break
        userdata.append(sentence+[0]*(maxlen-len(sentence)))
        if len(userdata)==maxsent:
            break
    user_query_data.append(userdata+[[0]*maxlen]*(maxsent-len(userdata)))


  if d == 'train':
    x_train = np.array(user_query_data, dtype = 'int32')
    y_train = np.array(to_categorical(user_label), dtype = 'float32') 
  elif d == 'test':
    x_test = np.array(user_query_data, dtype = 'int32')
    y_test = np.array(to_categorical(user_label), dtype = 'float32')
  elif d == 'val':
    x_val = np.array(user_query_data, dtype = 'int32')
    y_val = np.array(to_categorical(user_label), dtype = 'float32')







# In[8]:

'''
qdata = np.array(user_query_data,dtype='int32')
labels = np.array(to_categorical(user_label),dtype='float32')


indices = np.arange(len(qdata))
np.random.shuffle(indices)

nb_validation_samples=int(0.72*len(qdata))
x_train = qdata[indices[:int(0.72*len(qdata))]]
y_train = labels[indices[:int(0.72*len(qdata))]]


x_val = qdata[indices[int(0.72*len(qdata)):int(0.8*len(qdata))]]
y_val = labels[indices[int(0.72*len(qdata)):int(0.8*len(qdata))]]

x_test = qdata[indices[int(0.8*len(qdata)):]]
y_test = labels[indices[int(0.8*len(qdata)):]]

'''






# In[18]:

sentence_inputt = Input(shape=(maxlen,), dtype='int32')
embedding_layer = Embedding(len(word_dict), EMBEDDING_DIM, trainable=True)


embedded_sequencest = embedding_layer(sentence_inputt)
word_vec=Dropout(0.2)(embedded_sequencest)

cnn = Conv1D(filters=300, kernel_size=3,  padding='same', activation='relu', strides=1)(word_vec)

d_cnn=Dropout(0.2)(cnn)
w_dense = TimeDistributed(Dense(200,activation='tanh'), name='Dense')(d_cnn)

w_att = Flatten()(Activation('softmax')(Dense(1)(w_dense)))




sent_rep=Dot((1, 1))([d_cnn, w_att])
sentEncodert = Model(sentence_inputt, sent_rep)

userdata_input = Input(shape=(maxsent, maxlen), dtype='int32')
userdata_encoder = TimeDistributed(sentEncodert, name='sentEncodert')(userdata_input)

cnn_sent = Conv1D(filters=300, kernel_size=3, padding='same', activation='relu', strides=1)(userdata_encoder)
d_cnn_sent=Dropout(0.2)(cnn_sent)

s_dense = TimeDistributed(Dense(200,activation='tanh'), name='Dense')(d_cnn_sent)
s_att = Flatten()(Activation('softmax')(Dense(1)(s_dense)))
user_rep=Dot((1, 1))([d_cnn_sent, s_att])

preds = Dense(task_nodes_num, activation='softmax')(user_rep)#age Dense(6, activation='softmax')
model = Model([userdata_input], preds)

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

model.summary()






callbacks = [EarlyStopping(monitor='val_loss', patience = 3)]
model.fit([x_train], y_train, epochs=100, batch_size=100, validation_data = (x_val, y_val), callbacks = callbacks)  







for ep in range(1):
  print("ep = %s**************************"%(ep))
  model.fit([x_train], y_train, epochs=1, batch_size=100)  
  y_pred = model.predict([x_val], batch_size=256, verbose=1)
  y_pred = np.argmax(y_pred, axis=1)
  y_true = np.argmax(y_val, axis=1)
  print("VALIDATION Results/////////////////////")
  print(classification_report(y_true, y_pred,digits=4))
  print(accuracy_score(y_true, y_pred))
  y_pred_test = model.predict([x_test], batch_size=256, verbose=1)
  y_pred_test = np.argmax(y_pred_test, axis=1)
  y_test_true = np.argmax(y_test, axis=1)
  print("TEST Results/////////////////////////")
  print(classification_report(y_test_true, y_pred_test,digits=4))
  






y_pred = model.predict(x_test)
predicted = pd.DataFrame(y_pred).idxmax(axis = 1)
target = pd.DataFrame(y_test).idxmax(axis = 1)

print("weighted ##### acc:%f, pre:%f, rec:%f, f1:%f"%(accuracy_score(predicted, target), 
                                                      precision_score(predicted, target, average='weighted'),
                                                      recall_score(predicted, target, average='weighted'),
                                                      f1_score(predicted, target, average='weighted')
                                                      ))
print("macro ##### acc:%f, pre:%f, rec:%f, f1:%f"%(accuracy_score(predicted, target), 
                                                      precision_score(predicted, target, average='macro'),
                                                      recall_score(predicted, target, average='macro'),
                                                      f1_score(predicted, target, average='macro')
                                                      ))






test['predicted_gender'] = predicted
def gender_reverse(gender):
  if gender == 0:
    return 'f'
  else:
    return 'm'

test['gender_transfer'] = test['gender'].apply(gender_reverse)
test['predicted_gender_HURA'] = test['predicted_gender'].apply(gender_reverse)





All_gender_Results_all = pd.merge(All_gender_Results, test[['user_id', 'predicted_gender_HURA']], on = 'user_id', how = 'left')







print("weighted ##### acc:%f, pre:%f, rec:%f, f1:%f"%(accuracy_score(All_gender_Results_all['RF_app2vec_gender'],All_gender_Results_all['gender']), 
                                                      precision_score(All_gender_Results_all['RF_app2vec_gender'], All_gender_Results_all['gender'], average='weighted'),
                                                      recall_score(All_gender_Results_all['RF_app2vec_gender'],All_gender_Results_all['gender'], average='weighted'),
                                                      f1_score(All_gender_Results_all['RF_app2vec_gender'], All_gender_Results_all['gender'], average='weighted')
                                                      ))
print("macro ##### acc:%f, pre:%f, rec:%f, f1:%f"%(accuracy_score(All_gender_Results_all['RF_app2vec_gender'],All_gender_Results_all['gender']), 
                                                      precision_score(All_gender_Results_all['RF_app2vec_gender'], All_gender_Results_all['gender'], average='macro'),
                                                      recall_score(All_gender_Results_all['RF_app2vec_gender'], All_gender_Results_all['gender'], average='macro'),
                                                      f1_score(All_gender_Results_all['RF_app2vec_gender'], All_gender_Results_all['gender'], average='macro')
                                                      ))

