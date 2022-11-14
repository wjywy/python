# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 11:48:13 2022

@author: ASUS
"""

#先找基线(已解决，设定的阈值)；再切割，使用cut试一试
#思路就是准确将凸起的区间识别，然后将第一条和最后一条线做标注

import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont

# 创建一幅新图像用以绘制心电图形（白色背景）
im = Image.new('RGBA', (3000,1500), 'white')

# 创建一个画图实例
draw = ImageDraw.Draw(im)

#准备字体
font = ImageFont.truetype('simsun.ttc',50)
fontSpecial = ImageFont.truetype('simsun.ttc',50)

def redSoundHeart (file_name) :
    data = [] #存放心音数据
    file = open(file_name,'r')  #打开文件
    file_data = file.readlines() #读取所有行
    for rows in file_data :
        A = rows[:-1]
        n,d = A.split('   ') #分割之后，返回数组
        data.append({'value':int(float(d)*10)})
        dataSeries = pd.DataFrame(data) #存为DataFrame格式，好打标签
        dataSeries['overflow'] = dataSeries['value'].map(addCapture)
    data_overflow = dataSeries['overflow']
    for i in data_overflow :
        print(i)
    print(dataSeries)
    return dataSeries

#给数据打标签
def addCapture (data) :
    if(data > 300) :
        data = 1
        return data
    else :
        data = 0
        return data
    
def findMaxMin (data) :
    data_value = data['value'] #提取列
    data_overflow = data['overflow'] #提取标签列
    findMax = []
    maxNum = max(data_value)
    minNum = min(data_value)
    print(maxNum)
    print(minNum)
    i = 0
    numberArea = int(input('请输入采样频率：')) 
    numberHeart = numberArea*0.113  #由采样频率得出心率
    topAndBottom = len(data_value) / int(numberHeart) #计算几个波峰
    for count in data_overflow :
        findMax.append(count)
        i = i + 1
        if(i>int(numberHeart * 2)) : #一个周期为56次，则两个周期之中肯定会有一次波峰
            break
    find = max(findMax) 
    findcount = findMax.index(find)  #查找位置
    draw.text((100,100),'显示心音图',(255,0,0),font=font)#划线
    i = 0
    tmp = 0
    tmp1 = 0
    t = 0    #限制次数，防止索引超出
    w = 100
    for d in data_value:   
       tmp1 = 1200-(d- minNum)
       if d > 300 :
           draw.line([(w+i,tmp + tmp),(w+i+1,tmp1)],(64, 62, 65),2)#划线
           i = i + 1
           t = t + 1
           continue
       if i>1:
          draw.line([(w+i,tmp),(w+i+1,tmp1)],(0,255,255),2)#划线
       tmp = tmp1  
       i=i+1
    im.show()  #显示绘制的心音图像
    print(topAndBottom)
    print(numberHeart)
    print(data_overflow)
    print(findcount)
    print(find)
    
if __name__=="__main__":
    dataSeries = redSoundHeart('D:/心音软件ddd/UserData/Data/心音.txt')
    findMaxMin(dataSeries)

    
    
    
    
    
    
    
    
    