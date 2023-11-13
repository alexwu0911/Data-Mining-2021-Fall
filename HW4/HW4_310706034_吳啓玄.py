#!/usr/bin/env python
# coding: utf-8

# # 1. 資料前處理

# In[6]:


import pandas as pd
import numpy as np

df = pd.read_csv("./yelp.csv")
df = df[["stars","text"]]
df['stars'] = df['stars'].map(lambda x : 1 if x >=4 else 0)

from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import numpy as np

split_ratio = 0.8
split_point = int(split_ratio*len(df["text"]))
train_text = df["text"].tolist()[:split_point]
y_train = np.array(df["stars"].tolist()[:split_point])
test_text = df["text"].tolist()[split_point:]
y_test = np.array(df["stars"].tolist()[split_point:])

token = Tokenizer(num_words=10000) 
token.fit_on_texts(train_text)  

token.word_index #可以看到它將英文字轉為數字的結果，例如:the轉換成1

x_train_seq = token.texts_to_sequences(train_text)
x_test_seq = token.texts_to_sequences(test_text)

x_train = sequence.pad_sequences(x_train_seq, maxlen=2000)
x_test = sequence.pad_sequences(x_test_seq, maxlen=2000)


# # 2.1 CNN 建模
# ## a. 用CNN對train的資料進行建模，可自行設計神經網路的架構
# 
# ## b. 加入Dropout Layer設定Dropout參數(建議0.7)

# In[7]:


from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.regularizers import l2
import matplotlib.pyplot as plt

modelCNN = Sequential()
modelCNN.add(Embedding(output_dim=300, input_dim=10000, input_length=2000)) 
modelCNN.add(Conv1D(32, 3, activation='relu', padding="same", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
modelCNN.add(MaxPooling1D(3))
modelCNN.add(Conv1D(32, 3, activation='relu', padding="same", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
modelCNN.add(MaxPooling1D(3))
modelCNN.add(Flatten()) #需要
modelCNN.add(Dropout(0.7))
modelCNN.add(Dense(128, activation='relu'))
modelCNN.add(Dense(1, activation='sigmoid'))

print(modelCNN.summary())
modelCNN.compile(loss='binary_crossentropy',optimizer='Nadam',metrics=['accuracy'])
history = modelCNN.fit(x_train, y_train, epochs = 10, batch_size = 50, verbose = 1, validation_split = 0.2)


# ## c. plot出CNN訓練過程中的Accuracy與Loss值變化

# In[8]:


# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('CNN model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
#file_name = '/content/drive/MyDrive/DM HW4 picture/'+'CNN'+'accuracy.png'
#plt.savefig(file_name)
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('CNN model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
#file_name = '/content/drive/MyDrive/DM HW4 picture/'+'CNN'+'loss.png'
#plt.savefig(file_name)
plt.show()


# # 2.2 LSTM 建模
# ## a. 用LSTM 對train的資料進行建模，可自行設計神經網路的架構
# 
# ## b. 加入Dropout Layer設定Dropout參數(建議0.7)

# In[9]:


from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers import CuDNNLSTM
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from tensorflow import keras
from keras import optimizers
from tensorflow.keras import layers

modelLSTM = Sequential()
modelLSTM.add(Embedding(output_dim=300, input_dim=10000, input_length=2000)) 
modelLSTM.add(Dropout(0.2)) 
modelLSTM.add(CuDNNLSTM(32))
modelLSTM.add(Dense(units=256,activation='relu'))
modelLSTM.add(Dropout(0.2))
modelLSTM.add(Dense(units=1,activation='sigmoid'))

modelLSTM.summary()
adam = keras.optimizers.Adam() #default 0.001
modelLSTM.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
history = modelLSTM.fit(x_train, y_train, epochs = 10, batch_size = 50, verbose = 1, validation_split = 0.2)


# ## c. plot出LSTM訓練過程中的Accuracy與Loss值變化

# In[10]:


import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('LSTM model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
#file_name = '/content/drive/MyDrive/DM HW4 picture/'+data_set_name+'accuracy.png'
#plt.savefig(file_name)
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('LSTM model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
#file_name = '/content/drive/MyDrive/DM HW4 picture/'+data_set_name+'loss.png'
#plt.savefig(file_name)
plt.show()


# # 3.1 CNN模型評估(early stopping，訓練到epochs=3即停止，避免overfitting)
# 
# ## 利用test的資料對建立的CNN模型進行測試，並計算Accuracy

# In[12]:


modelCNN2 = Sequential()
modelCNN2.add(Embedding(output_dim=300, input_dim=10000, input_length=2000)) 
modelCNN2.add(Conv1D(32, 3, activation='relu', padding="same", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
modelCNN2.add(MaxPooling1D(3))
modelCNN2.add(Conv1D(32, 3, activation='relu', padding="same", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
modelCNN2.add(MaxPooling1D(3))
modelCNN2.add(Flatten()) #需要
modelCNN2.add(Dropout(0.7))
modelCNN2.add(Dense(128, activation='relu'))
modelCNN2.add(Dense(1, activation='sigmoid'))

print(modelCNN2.summary())
modelCNN2.compile(loss='binary_crossentropy',optimizer='Nadam',metrics=['accuracy'])
history = modelCNN2.fit(x_train, y_train, epochs = 3, batch_size = 50, verbose = 1, validation_split = 0.2)

CNN_scores = modelCNN2.evaluate(x_test, y_test,verbose=1)
print("CNN test set accuracy = ",round(CNN_scores[1],3))


# # 3.2 LSTM模型評估(early stopping，訓練到epochs=2即停止，避免overfitting)
# 
# ## 利用test的資料對建立的LSTM模型進行測試，並計算Accuracy

# In[14]:


modelLSTM2 = Sequential()
modelLSTM2.add(Embedding(output_dim=300, input_dim=10000, input_length=2000)) 
modelLSTM2.add(Dropout(0.2)) 
modelLSTM2.add(CuDNNLSTM(32))
modelLSTM2.add(Dense(units=256,activation='relu'))
modelLSTM2.add(Dropout(0.2))
modelLSTM2.add(Dense(units=1,activation='sigmoid'))

modelLSTM2.summary()
adam = keras.optimizers.Adam() #default 0.001
modelLSTM2.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
history = modelLSTM2.fit(x_train, y_train, epochs = 2, batch_size = 50, verbose = 1, validation_split = 0.2)

LSTM_scores = modelLSTM2.evaluate(x_test, y_test,verbose=1)
print("LSTM test set accuracy = ",round(LSTM_scores[1],3))

