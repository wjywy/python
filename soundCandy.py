# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 22:38:56 2022

@author: ASUS
"""
#选取数据进行属性分类，如分为年龄，身高，体脂率等。在对每一样属性进行划分
#阈值如年龄为10到20岁，20岁到30岁。然后统计每一个属性不同划分的频数，百
#分比等特征。
#预处理数据：填空与改错
#填空：1.忽略，删除这条不全的数据
#     2.人工填补（不推荐）
#      3.宏定义常量填补
#改错：对于数据异常的值进行如填补操作
#数据转换：把同一属性数据的不同划分进行编码为更适合的类型符号

#最后一个任务：将每个年龄段得相应并发症的概率求出来，还有：建立模型(决策树)
#可以根据数据对是否患上某种并发症进行决策树建模分析，然后我们再对表格进行数据分析，
#就可以得出患糖尿病的人群最有可能患上哪些并发症，最后就可以对患上糖尿病的做一个决策树

#建模：线性回归

import pandas as pd
import copy #导入深拷贝需要的包
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
from sklearn.impute import SimpleImputer   #导入IMputer包，处理缺失数据以后就用这个包
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
    cutNewData(newData,sort)
    newData = newData.sort_values(by=sort)
    print(newData)
    
    return newData

#切割数据进行定性分析
def cutNewData (data,sort) :
    maxSource = max(data[sort])
    minSource = min(data[sort])-1
    splitData = range(int(minSource),int(maxSource),int((maxSource-minSource)/7))
    splitArr = list(splitData)
    splitArr[-1] = int(maxSource)
   # print(splitArr)
    labels = ['small','lessSmall','middle','lessMiddle','big','lessBig','emptyBig'] #cut的标签必须比bins的边界少一个
    data[sort] = pd.cut(data[sort], splitArr,labels=labels)
    data.index = data[sort].tolist()
   # print(data)
    divide = ['small','lessSmall','middle','lessMiddle','big','lessBig','emptyBig']
    for eachData in divide :
        aboutSort = data.loc[[eachData]] #选取eachData的行
        calDiease(aboutSort)
    
#计算那种并发症出现概率大,顺便绘图可视化
def calDiease (data) :
    data = data.loc[:,'高血压':'颅内肿瘤'] #选择连续多列
    calCountToOne = []
    for eachData in data :
        countToOne = 0
        for everyData in data[eachData] :
            if everyData == 1 :
                countToOne = countToOne + 1
                continue
        calCountToOne.append((countToOne/2,eachData))
    calData = pd.DataFrame(calCountToOne).sort_values(0)
  #  print(calData)
    calData_x = calData[1].tolist()
    calData_y = calData[0].tolist()
    plt.figure(figsize=(40, 15)) 
    plt.plot(calData_x, calData_y)
    plt.ylabel('并发症比例（单位：%）')
    for x,y in zip(calData_x,calData_y) :
        plt.text(x, y, y)
    plt.show()

#对另一个表的数据分析加绘折线图
def readExcelToDiease (fileName) :
    data = pd.read_excel(fileName)
    copyData = copy.deepcopy(data)
    #统计农村和城市的患病率
    areaData = data['地区分类']
    areaData = areaData.tolist() #先将series数据转为list数据
    data.index = areaData #将地区分类作为索引
    area = ['一类农村','二类农村','三类农村','四类农村','大城市','中城市','小城市']
    getProbability = []
    #计算农村类别和城市类别分别的患病率
    for counAndCity in area :
        dataToArea = data.loc[[counAndCity]]
        Probability = 0
        for getDiease in dataToArea['患病率'] :
            Probability = (getDiease + Probability)
        getProbability.append((counAndCity,int(Probability)/4))
    DieaseProbability = pd.DataFrame(getProbability)
    # print(DieaseProbability) #根据地区患病比例
    #提取信息进行绘图
    area_x = DieaseProbability[0].tolist()
    area_y = DieaseProbability[1].tolist()
    plt.plot(area_x,area_y)
    plt.ylabel('患病比例（单位：%）')
    for x,y in zip(area_x,area_y):
        plt.text(x,y,y)
    plt.show()
    
    #统计按年龄分类的患病率
    ageData = data['结果指标']
    ageData = ageData.tolist()
    data.index = ageData
    age = ['35-44岁','45-54岁','55-64岁','65岁及以上']
    getAge = []
    for ageAll in age :
        dataToAge = data.loc[[ageAll]]
        ageProbability = 0
        for ageDiease in dataToAge['患病率'] : #年龄段的患病率 
            ageProbability = (ageDiease + ageProbability)
        getAge.append((ageAll,int((ageProbability)/8)))
    ageDieaseProbablity = pd.DataFrame(getAge)
    # print(ageDieaseProbablity)  #根据年龄区间患病比例
    #提取信息进行绘图
    age_x = ageDieaseProbablity[0].tolist()
    age_y = ageDieaseProbablity[1].tolist()
    plt.plot(age_x, age_y) 
    plt.ylabel('患病比例（单位：%）')
    for x,y in zip(age_x,age_y):
        plt.text(x,y,y)
    plt.show()
    
    #统计按年份分类的患病率
   # print(copyData)
    index = [7,35,63,85]
    Xdata = []
    for count in index :
        Xdata.append(copyData.loc[count,'患病率'])
   # print(Xdata)
    year_x = ['2008','2003','1998','1993']
    year_y = Xdata
    plt.plot(year_x, year_y)
    plt.ylabel('患病比例（单位：%）')
    for x,y in zip(year_x,year_y):
        plt.text(x,y,y)
    plt.show()
           
        
#控制变量，只改变一列数据的值，其他列数据保持不变,默认初始值为年龄，可自行输入 
if __name__ == "__main__" :
    newData = redSoundCandy("C:/Users/ASUS/Documents/Tencent Files/2934610933/FileRecv/糖尿病.xlsx")
    calDiease(newData)
    readExcelToDiease("C:/Users/ASUS/Documents/Tencent Files/2934610933/FileRecv/糖尿病患病率.xlsx")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    