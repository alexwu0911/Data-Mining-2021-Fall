#!/usr/bin/env python
# coding: utf-8

# ### 0.讀檔

# In[1]:


import pandas as pd

df = pd.read_csv("./新竹_2020.csv",encoding='big5')
df = df.drop(0) #刪除第一列
df


# ### 1. 資料前處理
# ##### (a) 取出10.11.12月資料

# In[2]:


for i in range(1,len(df)+1):
    month = df.loc[i].iat[1].split("/")[1] #loc取index
    if month not in ["10", "11", "12"]:
        df = df.drop(index=[i])
df


# ##### (b) 缺失值以及無效值以前後一小時平均值取代 (如果前一小時仍有空值，再取更前一小時)

# In[3]:


#先刪測站、日期
df = df.drop(df.columns[[0,1]], axis=1) #不用用到欄名的drop欄方法
df


# In[4]:


#轉置df成時間序列data frame: df_t
col_names = df.iloc[0:18].T.iloc[0].tolist() #df_t欄名
col_names = [i.strip() for i in col_names] #消除空白
row = [str(x) for x in range(24)]
df_t = df.iloc[0:18].T.loc[row]
df_t.columns = col_names #rename df_t colname

counter = 18
while counter < 1656:
    df_temp = df.iloc[counter:counter+18].T.loc[row]
    df_temp.columns = col_names #rename df_t colname
    df_t = df_t.append(df_temp)
    counter = counter+18

df_t #2208列是對的，24小時*92天=2208筆


# In[5]:


#對SO2最後一列補值=6.9
df_t.iloc[2207]["SO2"] = 6.9
df_t.iloc[2192:]


# In[6]:


#開始進行前後平均補值
miss_list = [] #遺失值樣貌
for i in range(0,len(df_t)):
    for j in range(0,18):
        try:
            float(df_t.iloc[i].iat[j]) 
        except: #有遺失值
            miss_list.append(df_t.iloc[i].iat[j])
            a=0; l=i-1
            while 1==1:#往前一小時找
                try:
                    a=float(df_t.iloc[l].iat[j])
                    break;
                except:
                    l=l-1
            b=0; l=i+1
            while 1==1:#往後一小時找
                try:
                    b=float(df_t.iloc[l].iat[j])
                    break;
                except:
                    l=l+1
            df_t.iloc[i].iat[j] = (a+b)/2 #前後一小時補值
                
                
df_t               


# In[8]:


#查看遺失值數量
from collections import Counter
Counter(miss_list)


# #### (c) 將資料切割成訓練集(10.11月)以及測試集(12月)

# In[9]:


train_df = df_t.iloc[0:1464]
test_df = df_t.iloc[1464:]


# ##### (d)製作時序資料: 將資料形式轉換為行(row)代表18種屬性，欄(column)代表逐時數據資料

# In[10]:


train_df = train_df.T
test_df = test_df.T
train_df


# ### 2. 時間序列

# ####  a.預測目標

# #####  1.  Y1: 將未來第一個小時當預測目標

# In[11]:


train_df_Y1 = [float(i) for i in train_df.iloc[9].tolist()[6:]]
test_df_Y1 = [float(i) for i in test_df.iloc[9].tolist()[6:]]


# #####  2.  Y2: 將未來第六個小時當預測目標

# In[12]:


train_df_Y2 = [float(i) for i in train_df.iloc[9].tolist()[11:]]
test_df_Y2 = [float(i) for i in test_df.iloc[9].tolist()[11:]]


# ####  b. X分別取

# ##### 1. X1: 只有PM2.5 (e.g. X[0]會有6個特徵，即第0~5小時的PM2.5數值)

# In[13]:


train_df_X1= []
for i in range(0,1458):
    train_df_X1.append([float(i) for i in train_df.iloc[9].tolist()[i:i+6]])
    
test_df_X1= []
for i in range(0,738):
    test_df_X1.append([float(i) for i in test_df.iloc[9].tolist()[i:i+6]])


# ##### 2. X2: 所有18種屬性 (e.g. X[0]會有18*6個特徵，即第0~5小時的所有18種屬性數值)

# In[14]:


train_df_X2= []
for i in range(0,1458):
    temp = []
    for j in range(0,18):
        temp = temp+[float(i) for i in train_df.iloc[j].tolist()[i:i+6]]
    train_df_X2.append(temp)
    
test_df_X2= []
for i in range(0,738):
    temp = []
    for j in range(0,18):
        temp = temp+[float(i) for i in test_df.iloc[j].tolist()[i:i+6]]
    test_df_X2.append(temp)


# ####  c. 使用兩種模型 Linear Regression 和 XGBoost 建模

# In[15]:


import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

#########################Build model###############################
lm_model = LinearRegression()
xgboostModel = XGBRegressor(n_estimators=1000, learning_rate= 0.3)
#########################linear regression#########################

######################### model=lm #########################
#x=X1, y=Y1, model=lm
lm_model.fit(train_df_X1, train_df_Y1)
yfit1 = lm_model.predict(test_df_X1)

#x=X1, y=Y2, model=lm
lm_model.fit(train_df_X1[:1453], train_df_Y2)
yfit2 = lm_model.predict(test_df_X1[:733])

#x=X2, y=Y1, model=lm
lm_model.fit(train_df_X2, train_df_Y1)
yfit3 = lm_model.predict(test_df_X2)

#x=X2, y=Y2, model=lm
lm_model.fit(train_df_X2[:1453], train_df_Y2)
yfit4 = lm_model.predict(test_df_X2[:733])

######################### model=xgboost #########################
#x=X1, y=Y1, model=xgboost
xgboostModel.fit(np.array(train_df_X1), np.array(train_df_Y1))
yfit5 = xgboostModel.predict(np.array(test_df_X1))

#x=X1, y=Y2, model=xgboost
xgboostModel.fit(np.array(train_df_X1[:1453]), np.array(train_df_Y2))
yfit6 = xgboostModel.predict(np.array(test_df_X1[:733]))

#x=X2, y=Y1, model=xgboost
xgboostModel.fit(np.array(train_df_X2), np.array(train_df_Y1))
yfit7 = xgboostModel.predict(np.array(test_df_X2))

#x=X2, y=Y2, model=xgboost
xgboostModel.fit(np.array(train_df_X2[:1453]), np.array(train_df_Y2))
yfit8 = xgboostModel.predict(np.array(test_df_X2[:733]))


# ##### d. 用測試集資料計算MAE (會有8個結果， 2種X資料 * 2種Y資料 * 2種模型)

# In[16]:


print("x=X1, y=Y1, model=lm MAE:",mean_absolute_error(test_df_Y1,yfit1))
print("x=X1, y=Y2, model=lm MAE:",mean_absolute_error(test_df_Y2,yfit2))
print("x=X2, y=Y1, model=lm MAE:",mean_absolute_error(test_df_Y1,yfit3))
print("x=X2, y=Y2, model=lm MAE:",mean_absolute_error(test_df_Y2,yfit4))
print("x=X1, y=Y1, model=xgboost MAE:",mean_absolute_error(test_df_Y1,yfit5))
print("x=X1, y=Y2, model=xgboost MAE:",mean_absolute_error(test_df_Y2,yfit6))
print("x=X2, y=Y1, model=xgboost MAE:",mean_absolute_error(test_df_Y1,yfit7))
print("x=X2, y=Y2, model=xgboost MAE:",mean_absolute_error(test_df_Y2,yfit8))

