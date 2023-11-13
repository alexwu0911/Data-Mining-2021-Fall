#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
#1. 資料前處理
#a. 讀取csv檔僅保留"text"、"stars"兩個欄位，並將stars欄位內值大於等於4的轉成1，其餘轉成0，1: positive; 0: negative
df = pd.read_csv("./yelp.csv")
df = df[["stars","text"]]
df['stars'] = df['stars'].map(lambda x : 1 if x >=4 else 0)


# In[2]:


#b.去除停頓詞stop words 
import nltk
from sklearn.feature_extraction.text import CountVectorizer

corpus = df["text"].tolist()
nltk.download('stopwords')
nltk_stopwords = nltk.corpus.stopwords.words('english')
stop_words = nltk_stopwords

#c.文字探勘前處理，將文字轉換成向量，實作 tf-idf
vectorizer  = CountVectorizer(stop_words=stop_words, min_df=0.01)
X = vectorizer.fit_transform(corpus)
features = vectorizer.get_feature_names()

from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(X)
tfidf_vec = tfidf.toarray()
tfidf_df = pd.DataFrame(tfidf_vec)
tfidf_df['y'] = df['stars']
tfidf_df = tfidf_df.dropna()
tfidf_df


# In[6]:


#c.文字探勘前處理，將文字轉換成向量，實作 word2vec
#將corpus分割並儲存到text_seg_list(需要一些時間)
text_seg_binary_list = X.toarray()
text_seg_list = []
for i in range(len(text_seg_binary_list)):
    temp_list = []
    for j in range(len(text_seg_binary_list[i])):
        if text_seg_binary_list[i][j]==1:
            temp_list.append(features[j])
    text_seg_list.append(temp_list)

#訓練Word2Vec
from gensim.models import Word2Vec
vector_size=250
model = Word2Vec(sentences=text_seg_list, vector_size=vector_size, epochs=10)

#word embedding
import numpy as np
word2vec_vec = [] 
for i in range(len(text_seg_list)):
    vector_sum = np.zeros(vector_size)
    count = 0
    for j in range(len(text_seg_list[i])):
        try:
            vector_sum = vector_sum+model.wv[text_seg_list[i][j]]
            count += 1
        except: #該字沒有vector
            pass
    vector_average = vector_sum/count
    word2vec_vec.append(vector_average.tolist())

word2vec_df = pd.DataFrame(word2vec_vec)
word2vec_df['y'] = df['stars']
word2vec_df = word2vec_df.dropna()
word2vec_df


# In[20]:


#2.
#K fold cv function and random forest
import random
from sklearn import ensemble

def K_fold_CV(k, data):
    partition = []
    partition_index = []
    data_num = len(data)    
    for i in range(k):
        if i != k-1:
            partition.append(data_num//k)
        else:
            partition.append(data_num//k+data_num%k)
        partition_index.append([sum(partition)-partition[i],sum(partition)-1])
       
    random.seed(123)
    data_df_shuffled = data.sample(frac=1).reset_index(drop=True) #data shuffle
    Accuracy = 0
    for i in range(k):
        print("目前test fold=",str(i+1),end=' ')
        test_x = data_df_shuffled.iloc[partition_index[i][0]:partition_index[i][1]+1].drop("y", axis = 1)
        test_y = data_df_shuffled.iloc[partition_index[i][0]:partition_index[i][1]+1]["y"]
        train_x = data_df_shuffled.drop(list(range(partition_index[i][0], partition_index[i][1]+1))).drop("y", axis = 1)
        train_y = data_df_shuffled.drop(list(range(partition_index[i][0], partition_index[i][1]+1)))["y"]

        forest_fit = forest.fit(train_x, train_y)
        test_y_predicted = forest.predict(test_x)
        
        count = 0
        for j in range(len(test_y)):
            if(test_y.iloc[j]==test_y_predicted[j]):
                count = count + 1
        print(", accuracy=",str(round((count/len(test_y)),4)))
        Accuracy = Accuracy + count/len(test_y)
        
    print("---------------------------------------")
    print("Average accuracy:")
    return(round(Accuracy/k,4))    


# In[22]:


# 建立 random forest 模型
forest = ensemble.RandomForestClassifier(n_estimators = 200, criterion="entropy")
# tfidf_df 4-fold cv
K_fold_CV(4, tfidf_df)


# In[23]:


# 建立 random forest 模型
forest = ensemble.RandomForestClassifier(n_estimators = 200, criterion="entropy")
# word2vec 4-fold cv
K_fold_CV(4, word2vec_df)

