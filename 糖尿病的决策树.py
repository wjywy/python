# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 13:24:27 2022

@author: ASUS
"""
#1为正常心音，0为非正常心音

import numpy as np
import pandas as pd
from sklearn import tree  
from sklearn.tree import export_graphviz  
import graphviz
from PIL import Image,ImageDraw,ImageFont
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False #正常显示中文与负号

#在这里处理每个心音文件都要进行的操作
def ReadNormalFile(fileName) :
    # sep代表每一行的分隔符，names表示类名
    data = pd.read_table(fileName,sep='   ',engine=('python'),names=['索引','幅度'])
    data = data.drop('索引',axis=1)  #此方法不会改变原数据 
    xData = data.index.tolist()
    yData = data['幅度'].tolist()
    # dataCopy = data['幅度']
    plt.plot(xData, yData)
    plt.ylabel('幅度')
    plt.show()
    dataMean.append(np.mean(yData))
    dataMax.append(max(yData)) 
    dataMin.append(min(yData))      
    dataVar.append(np.var(yData))    
    dataMedian.append(np.median(yData))   
    dataCov.append(np.cov(yData))
    dataNormal.append(1)        
    # print(dataVar)
    
#创建为一个dataFrame表
def changeDataFrame() :
    data = [dataMean,dataMax,dataMin,dataVar,dataMedian,dataCov,dataNormal]
    # print(data)
    data = pd.DataFrame(data)
    data = data.T
    data.index = ['正常心音']*len(data.index)
    data.columns = label
    print(data.values)
    plt.scatter()
    print(data)
    print(dataMean)
    return data
    
# 导出为excel表格，名称使用一次之后就不能重复使用，如test.xlsx
# def putExcel(data) :
#     with pd.ExcelWriter('test1.xlsx') as writer:
#         data.to_excel(writer, sheet_name='data')
    
    
if __name__ == "__main__" :
    fileArr = ['D:/心音软件ddd/UserData/Data/001正常心音a.txt',
               'D:/心音软件ddd/UserData/Data/001正常心音2a.txt',
               'D:/心音软件ddd/UserData/Data/001正常心音3a.txt',
               'D:/心音软件ddd/UserData/Data/001正常心音4a.txt']
    dataMean = []  #均值
    dataMax = []   #最大值
    dataMin = []     #最小值
    dataVar = []    #方差
    dataMedian = []  #标准差
    dataCov = []       #计算众数
    dataNormal = []    #标签判断是否正常，人工添加
    label = ['均值','最大值','最小值','方差','标准差','众数','是否正常']
    for count in range(len(fileArr)) :
        ReadNormalFile(fileArr[count])
    data = changeDataFrame()
    # putExcel(data)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        