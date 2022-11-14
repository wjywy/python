# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 17:21:01 2022

@author: ASUS
"""
#DataFrame.sample(n=none,frac=None,replace=False,weights=None,random_state=None
#                 axis=None )
#    n是要抽取的行数
#    frac是抽取的比例
#    replace是否有放回抽样，取replace==ture时为有放回抽样
#    weights是每个样本的权重
#    axis==0时抽取行，axis==1抽取列

import pandas as pd
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
from sklearn import linear_model  #导入线性模型

#返回一个进行处理后的DataFrame
def redSoundCandy(fileName,sort = '年龄'):
    data = pd.read_excel(fileName)
    pf = pd.DataFrame(data) #将数据转换为DataFrame格式
    arrData = [] #将不全为空的array装进数组
    for eachData in pf :
        everyData = pf[eachData] #循环对每一列进行处理
        junzhi = pf[eachData].mean() #对每一列求均值
        replaceNaN = everyData.fillna(junzhi) #用每列的平均值填补每一列
        if eachData == sort :
            sortData = replaceNaN.sort_values() #对替换后的数据进行升序排列
            replaceNaN = sortData
#对每一列数据进行检测是否全为空
        if replaceNaN.isnull().all() == True :
            del replaceNaN 
        else : 
            arrData.append(replaceNaN)
            continue
    newData = pd.DataFrame(arrData)
    newData = newData.T #进行转置
    newData = newData.sort_values(by=sort)
    print(newData) 
    BMIData = newData['BMI']   #提取BMI指数
    xTrain = BMIData.sample(frac=0.8,axis=0)   #随机提取80%的数据作为训练集
    xTrain = pd.DataFrame(xTrain)    #转为二维数据
    xTest = BMIData.sample(frac=0.8,axis=0)    #随机提取20%的数据作为测试集
    xTest = pd.DataFrame(xTest)      #转为二维数据
    # 切分答案
    columnData = newData.loc[:,'高血压':'颅内肿瘤']
    labelData = columnData.columns.tolist()
    for label in labelData :
        sigalData = columnData[label] 
        yTrain = sigalData.sample(frac=0.8,axis=0)
        yTrain = pd.DataFrame(yTrain)
        print(yTrain)
        yTest = sigalData.sample(frac=0.8,axis=0)
        yTest = pd.DataFrame(yTest)
        print(yTest)
        #研究计算
# linear_model.linearRegresssion()
#     主要参数：fit_intercept:布尔值，默认为true，若参数值为true时，代表训练模型需要加一个
#                           截距项；若参数为false时，代表模型无需加截距项
#             normalize：布尔值，默认为false，若fit_intercept为fasle时，normalsize参数
#                       无需设置；若normalize参数为true时，则输入的样本数据将(x-x均值)/|x|
#                       ,若设置normalsize为false时，在训练模型前，可以使用sklearn.preprocessing.StandardScaler
#                       进行标准化处理
#             copy_X:布尔值，默认为true。是否对X进行复制，如果选择false，则直接对原数据进行
#                    覆盖。（及经过中心化、归一化后，是否把新数据覆盖到原数据上），true则赋值X
#             n_jobs:整型，默认为一。计算时设置的任务个数。如果选择-1则代表使用所有的CPU。
#                    这一参数的对于目标个数>1且足够大规模的问题有加速作用
#    返回值：coef_:数组型变量，形状为(n_features)或(n_target,n_features)。对于线性回归问题
#                计算得出的feature的系数，即权重向量。如果输入的是多目标问题，则返回一个二维数组(n_target,n_feature)
#                如果是单目标问题，则返回一个一维数组，即n_features 
#          intercept:数组型变量。线性模型中的独立项，即b值
#    方法：desicion_function(X):对训练数据X进行预测
#         fit(X,y[,n_jobs]):对训练集X，y进行训练。是对scipy.linalg.lstsq的封装
#         get_params([deep]):得到该估计器的参数
#         predict(X):使用训练得到的估计器对输入为X的集合进行预测(X可以是测试集，也可以是需要预测的数据) 
#         score(X, y[,]sample_weight) ：返回对于以X为samples，以y为target的预测效果评分
#         set_params(params)：设置估计器的参数     
        regr = linear_model.LinearRegression()  #创建线性回归
        regr.fit(xTrain, yTrain)
        print('cofficients:\n',regr.coef_)   #系数
        # 绘图
        plt.scatter(xTest, yTest, color='black')
        plt.plot(xTest, regr.predict(xTest),color='blue',linewidth=3)
        plt.ylabel(label)
        plt.xticks(())
        plt.yticks(())
        plt.show()
        

if __name__ == "__main__" :
    newData = redSoundCandy("C:/Users/ASUS/Documents/Tencent Files/2934610933/FileRecv/糖尿病.xlsx")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    