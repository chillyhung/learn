#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy
import pandas as pd
from sklearn import preprocessing
numpy.random.seed(10)


# In[2]:


all_df = pd.read_excel("titanic3.xls")


# In[3]:


all_df[:2]


# In[4]:


#只選取需要的欄位
cols=['survived','name','pclass','sex','age','sibsp','parch','fare','embarked']


# In[5]:


all_df=all_df[cols]


# In[6]:


#偷看一下
all_df[:3]


# In[7]:


#依照8:2比例將資料分為訓練與測試
msk = numpy.random.rand(len(all_df))<0.8
train_df = all_df[msk]
test_df = all_df[~msk]


# In[8]:


print('總共:',len(all_df),',train:',len(train_df),',test:',len(test_df))


# 將訓練資料與測試資料進行預處理

# In[18]:


#定義preprocessdata
def preprocessdata(all_df):
    df=all_df.drop(['name'], axis =1 )
    
    age_mean = df['age'].mean()
    df['age'] = df['age'].fillna(age_mean)

    fare_mean = df['fare'].mean()
    df['fare'] = df['fare'].fillna(fare_mean)
    
    df['sex']= df['sex'].map({'female':0, 'male': 1}).astype(int)
    
    x_OneHot_df = pd.get_dummies(data=df,columns=["embarked" ])
    
    ndarray = x_OneHot_df.values
    Label = ndarray[:,0]
    Features = ndarray[:,1:]
    
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaledFeatures=minmax_scale.fit_transform(Features)
    
    return scaledFeatures,Label


# 將訓練資料與測試資料進行預處理

# In[21]:


#train_features(訓練資料的特徵欄位),train_Label(訓練資料的標籤欄位)
#test_features(訓練資料的特徵欄位),test_Label(訓練資料的標籤欄位)
train_features,train_Label=preprocessdata(train_df)
test_features,test_Label=preprocessdata(test_df)

print(train_features)


# #train_features(訓練資料的特徵欄位),train_Label(訓練資料的標籤欄位)
# #test_features(訓練資料的特徵欄位),test_Label(訓練資料的標籤欄位)
# train_features,train_Label=train_df
# test_features,test_Label=test_df

# ///開始建立模型
# ----------

# In[20]:


# 導入函式庫
import numpy as np  
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils  # 用來後續將 label 標籤轉為 one-hot-encoding  
from matplotlib import pyplot as plt


# In[12]:


# 建立簡單的線性執行的模型
model = Sequential()

# 建立輸入層與隱藏層1
model.add(Dense(units=40, 
                input_dim=9, kernel_initializer='uniform', activation='relu')) 

#units=40,輸出是隱藏層1,共有40個神經元
#input_dim=9, 9個特徵值
#kernel_initializer='normal'  使用uniform distribution分布的亂數,初始化weight與bias
#activation='relu'   定義激活函數relu


# In[13]:


# 建立隱藏層2
model.add(Dense(units=30, 
                kernel_initializer='uniform', activation='relu')) 


# In[14]:


# 建立輸出層
model.add(Dense(units=1, 
                kernel_initializer='uniform', activation='sigmoid')) 

# units=1  輸出層共有1個神經元
#kernel_initializer='normal'  使用uniform distribution分布的亂數,初始化weight與bias
#activation='relu'   定義?????


# //開始訓練
# --------

# In[15]:


#定義訓練方式

model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])
#optimizer 優化方法 使訓練加快收斂 提高準確率
#metrics 設定評估模型方式是accuracy準確率


# In[16]:


#開始訓練

train_history = model.fit(x=train_features, 
                          y=train_Label, 
                          validation_split=0.1, 
                          epochs=30, 
                          
                          batch_size=30 , verbose=2)

#使用model.fit進行訓練,訓練過程會存在train_history變數
#x=train_features 共有9特徵值
#y=train_Label    Label標籤欄位(是否生存? 生存=1)
#90%為訓練資料 10%為驗證資料

#epochs=30 執行30次訓練週期
#batch_size=30 每一批次30筆資料
##意思是:每一epochs(訓練週期),使用930筆資料進行訓練,分為每一批次30筆,所以大約分為31批次(930/30=31),進行訓練
##此epochs訓練完成後,計算此epochs訓練後的accuracy&loss
                          
#verbose=2 顯示訓練過程
#以下資訊
#90%為訓練資料 10%為驗證資料----->Train on 930 samples, validate on 104 samples
#使用訓練資料,計算loss與準確率------>loss: 9.3597
#使用驗證資料,計算loss與準確率------>val_loss: 12.5700


# In[17]:


#show_train_history(train_history,'acc','val_acc')

plt.plot(train_history.history['acc'])  
plt.plot(train_history.history['val_acc'])  
plt.title('Train History')  
plt.ylabel('acc')  
plt.xlabel('Epoch')  
plt.legend(['acc', 'val_acc'], loc='upper left')  
plt.show()
#
plt.plot(train_history.history['loss'])  
plt.plot(train_history.history['val_loss'])  
plt.title('Train History')  
plt.ylabel('loss')  
plt.xlabel('Epoch')  
plt.legend(['loss', 'val_loss'], loc='upper left')  
plt.show() 


# In[ ]:




