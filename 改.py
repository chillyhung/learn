#!/usr/bin/env python
# coding: utf-8

# In[2]:


#真的麻煩你了 我自己嘗試很久卡了很久

#代碼說明
#T13111檔案中有3筆資料:X、V、Y，其畫成座標值(X,Y)、(V,Y)，如下圖藍點綠點所示
#有一函數為(1+np.erf(np.log(x/a)/(b*np.sqrt(2))))*(1+np.erf(np.log(v/c)/(d*np.sqrt(2))))*0.25
#其中的變數有abcd

#簡單的來說就是有座標(X,Y)、(V,Y) 有abcd的答案 希望能透過機器學習 未來給新的座標也能得到abcd


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from sympy import integrate,erf,exp
import math
from pandas import read_csv
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler


# 待擬合的數據
# 要改成讀取雨量崩塌資料CSV檔

filename = 'T13111.csv'
names = ['I1', 'R1', 'class']
data = read_csv(filename, names=names)

#數據分成XY
array = data.values
X = array[:, 0]
v = array[:, 1]
Y = array[:, 2]

#--------------------------------------------------------------------------------------------------
    
# 易損性vulnerability
def vulnerability(p,x,v):
    a, b ,c ,d= p
    return (1+np.erf(np.log(x/a)/(b*np.sqrt(2))))*(1+np.erf(np.log(v/c)/(d*np.sqrt(2))))*0.25
    

def error(p, x, v, y):
    return vulnerability(p, x, v) - y


# 對參數求解
def slovePara():
    p0 = [0.14, 0.72, 0.69,100]        # p0為定義起始的函數,不會影響結果,影響找到最佳解時間

    global v
    Para = leastsq(error, p0, args=(X, v, Y))
    return Para

#-------------------------------------------------------------------------------------------------
#這區是使用最小平方法(leastqu)去擬合函數找出abcd這4個值
#我們老師目前想改掉最小平方法改以類神經的方法，但不知道是不是可以換成model = Sequential()這種簡單的建立模型，或是其他模型
#如果是換成model = Sequential()這個模型的話我一直想不通要怎麼去改，就是輸入輸出的部分


# 输出最后的结果
def solution():
    Para = slovePara() 
    a, b, c, d = Para[0]  
    print ("a=",a," b=",b,"c=",c,"d=",d)
    print ("cost:" + str(Para[1]))


    
    
    plt.figure(figsize=(8,8))
    global v
    plt.scatter(v, Y, color="blue", label="sample data2", linewidth=3)
    plt.scatter(X, Y, color="green", label="sample data1", linewidth=0.5)
    
    
    #   畫線
    x=np.linspace(0,1.5,100) ##在0-3直接畫100個連續點
    v=np.linspace(0,1.5,100) ##在0-3直接畫100個連續點
    y=(1+np.erf(np.log(x/a)/(b*np.sqrt(2))))*(1+np.erf(np.log(v/a)/(b*np.sqrt(2))))*0.25 ##函數式
    plt.plot(x,y,color="red",label="solution line",linewidth=2)
    plt.legend() #绘制图例
    plt.show()


solution()

                         


# In[ ]:




