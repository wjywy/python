# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 13:24:27 2022

@author: ASUS
"""

import numpy as np
import pandas as pd
from sklearn import tree  
from sklearn.tree import export_graphviz  
import graphviz
from PIL import Image,ImageDraw,ImageFont
from matplotlib import pyplot as plt
plt.rcParams['axes.unicode_minus'] = False #正常显示中文与负号

#在这里处理每个心音文件都要进行的操作
def basicReadFile(fileName) :
    # sep代表每一行的分隔符，names表示类名
    data = pd.read_table(fileName,sep='   ',engine=('python'),names=['索引','幅度'])
    data = data.drop('索引',axis=1)  #此方法不会改变原数据 
    xData = data.index.tolist()
    yData = data['幅度'].tolist()
    plt.plot(xData, yData)
    plt.ylabel('幅度')
    plt.show()
    
    
if __name__ == "__main__" :
    fileArr = ['D:/心音软件ddd/UserData/Data/心音.txt',
               'D:/心音软件ddd/UserData/Data/心音2.txt',
               'D:/心音软件ddd/UserData/Data/心音3.txt',
               'D:/心音软件ddd/UserData/Data/心音4.txt',
               'D:/心音软件ddd/UserData/Data/心音5.txt',
               'D:/心音软件ddd/UserData/Data/心音6.txt',
               'D:/心音软件ddd/UserData/Data/心音7.txt']
    for count in range(len(fileArr)) :
        basicReadFile(fileArr[count])