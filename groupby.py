# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 19:31:44 2022

@author: ASUS
"""

#将数据拆分为组
import pandas as pd 
import pandas as pd

ipl_data = {'Team': ['Riders', 'Riders', 'Devils', 'Devils', 'Kings',
         'kings', 'Kings', 'Kings', 'Riders', 'Royals', 'Royals', 'Riders'],
         'Rank': [1, 2, 2, 3, 3,4 ,1 ,1,2 , 4,1,2],
         'Year': [2014,2015,2014,2015,2014,2015,2016,2017,2016,2014,2015,2017],
         'Points':[876,789,863,673,741,812,756,788,694,701,804,690]}
df = pd.DataFrame(ipl_data)
print (df.groupby('Team'))   #已将数据分组，但是显示不出来相关数据
print(df.groupby('Team').groups)  #可以查看分组
print(df.groupby(['Team','Year']).groups)  #按多列分组   

