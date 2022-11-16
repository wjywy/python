# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 21:18:19 2022

@author: ASUS
"""

# #决策树是树状结构，它的每个叶节点对应着一个分类，非叶节点对应着在某个属性上的划分，根据样本在
# # 在该属性上的不同取值将其划分为若干子集。构造决策树的核心问题是在每一步如何选择恰当的属性
# # 对样本做拆分。ID3使用信息增益作为属性选择度量，C4.5使用增益率进行属性选择度量，CART使用基尼系数。


# # ID3算法
# # 输入：训练元组和他们对应的序列标号
# # 输出：决策树
# # 方法：1.对当前样本集合，计算所有属性的信息增益
#      # 2.选择信息增益最大的属性作为测试属性，把测试属性取值相同的样本划分为同一个样本集
#      # 3.若子样本集的类别属性只含有单个属性，则分支为叶子结点，判断其属性值并标记相应的标号
#      # 后返回调用处，否则对子样本集递归调用本算法
# from sklearn.datasets import load_iris  #鸢尾花数据集
# import pandas as pd
# from sklearn import tree   #决策树
# from sklearn.tree import export_graphviz  #  
# import graphviz       #生成决策树的pdf
# iris = load_iris()    
# # DecisonTreeClassifier参数：
# #    criterion:特征选择参数，默认gini，即CART算法
# #    splitter:特征划分标准：[best,random],best在特征的所有划分点中找出最优的划分点，random随机的在部分划分点中找局部最优的划分点
# #                         默认的best适合样本量不大的时候，而如果样本量非常大，此时决策树推荐random
# #    max_depth:决策树最大深度，默认值是'none'。当数据比较少或者特征比较少的时候可以不考虑。如果模型样本数量多
# #              特征也多时，推荐限制这个最大深度，具体取值取决于数据的分布。常用的可以取值10-100之间，常用来解决过拟合
# #    min_samples_split：内部节点再划分所需最小样本数。默认值为2。如果是int，则取传入值本身作为最小样本数。
# #                      如果是float，则取ceil(min_samples_split*样本数量)作为最小样本数
# #    min_samples_leaf:叶子节点最小样本数。如果是int，则取传入值本身作为最小样本数，如果是float，则取
# #                     ceil(min_samples_leaf*样本数量)的值作为最小样本数。这个值限制了叶子节点最小的样本数，如果某
# #                     叶子节点数目小于样本数，则会和兄弟节点一起被剪枝
# #    min_weight_fraction_leaf:叶子节点最小的样本权重和，默认为0.这个值限制了叶子节点所有样本权重和的最小值，如果
# #                             小于这个值，则会和兄弟节点一起被剪枝
# #                             默认为0，就是不来率权重问题，所有样本的权重相同。
# #    max_feature:在划分数据集时考虑的最多特征值数量。int值，在每次split时最大特征数；float值表示百分数
# #    max_leaf_nides:最大叶子节点数。默认为none
# #                   通过设置最大叶子节点数，可以防止过拟合，默认情况下是不设置最大叶子节点数
# #                   如果特征不多，可以不考虑这个值
# #    min_impurity_decrease:节点划分最小不纯度，默认值为‘0’
# #                          限制决策树的增长，如果某节点的不纯度小于这个阈值，则该节点不再生成子节点 
# #    min_impurity_split:信息增益的阈值。决策树在创建分支时，信息增益必须大于这个阈值，否则不分裂
# #    class_weight:类别权重，默认为None.指定样本个列名的权重，主要是为了防止训练集某些类别的过多，导致训练的决策树过于偏向这些类型
# clf = tree.DecisionTreeClassifier() 
# #常用方法：.fit——训练模型   .predict——用模型进行预测，返回预测值    
# #        .predict_log_proba()  返回一个数组，数组的元素依次是x预测为各个类别的概率的对数值
# #        .predict_proba()   返回一个数组，数组的元素依次是x预测为各个类别的概率的值 
# #        .score    返回模型的性能得分
# clf = clf.fit(iris.data, iris.target)     
# dot_file = 'tree.dot'
# tree.export_graphviz(clf,out_file=dot_file)
# with open("D:\\tree.dot","w") as f :
#     f = export_graphviz(clf,out_file=f,feature_names=['SL','SW','PL','PW'])


# # KNN算法
# # K-最近邻算法根据距离函数计算待分类样本x和每个训练样本的距离（作为相似度），选择待分与待分类
# # 样本距离最小的K个样本作为x的K个最近邻，最后以x的K个最近邻中的大多数样本所属的类别作为x的
# # 类别
# # 如何度量样本之间的距离是KNN算法的关键步骤之一，常见的相似度量方法包括闵可夫斯基距离(当
# # 参数p=2时为欧几里得距离，当p=1时为曼哈顿距离)、余弦相似度、皮尔逊相似系数、汉明距离、杰卡德
# # 相似系数等
# # 输入：簇的数目K和包含n个对象的数据库
# # 输出：K个簇，使平方误差最小
# # 方法：1.初始化距离为最大值
#       # 2.计算测试样本与每个训练样本之间的距离dist
#       # 得到目前K个最近邻样本中的最大距离maxdist
#       # 如果dist小于maxdist，则将该训练样本作为K最近邻样本
#       # 重复步骤(2)--(4)，直到测试样本和所有训练样本的距离都计算完毕
#       # 统计K个最近邻样本中每个类别出现的次数
#       # 选择出现频率最高的类别作为测试样本的类别
# # 警告：如果发现两个邻居，邻居k+1和邻居k具有相同距离但是不同的标签，则结果取决于训练数据的排序
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap   #导入颜色表
# from sklearn.neighbors import KNeighborsClassifier   #导入最近邻算法
# from sklearn.datasets import load_iris   #导入鸢尾花数据集
# iris = load_iris()
# X = iris.data[:,:2]    #截取数据的第三列还是第二列，记不清了，应该是第三列
# Y = iris.target        #样本的目标属性
# print(iris.feature_names)
# cmap_light = ListedColormap(['#FFAAAA','#AAFFAA','#AAAAFF'])
# cmap_bold = ListedColormap(['#FF0000','#00FF00','#0000FF'])
# # KNeighborsClassifier的参数： 
# #     n_neighbors：默认值为5，表示kneighbors查询使用的邻居数，就是K-NN的k的值，选取最近的k个点
# #     weights：默认为uniform，参数可以是uniform、distance，也可以是用户自己自定义的函数。uniform是
# #             均等的权重，就是说所有的邻近点的权重是相等的，distance是不均等的权重，距离近的点比距离圆远的点的影响大。
# #             用户自定义的函数。接收距离的数组，返回一组维数相同的权重。
# #     algorithm：快速k近邻搜索算法，默认参数为auto，可以理解为算法自己决定合适的搜索算法。用户也可以自己指定搜索
# #               算法，ball_tree,kd_tree,brute方法进行搜索，brute是暴力扫描，当数据很大时，非常耗时。kd_tree，
# #               构造kd数存储数据以便对其进行快速检索的树形数据结构，kd树也就是数据结构中的二叉树。
# #               以中值切分构造的树，每个节点是一个超矩形，在维数小于20时效率高。ball_tree是为了克服kd_tree在
# #               纬度过高时失效的问题，其构造过程是以质心和半径r分割样本空间，每个节点是一个超球体
# #    leaf_size：默认是30，这个是构造的kd树和ball树的大小。这个值的设置会影响树构建的速度和搜索速度，同样也影响着
# #              存储树所需要的内存大小。需要根据问题的性质选择最优的大小
# #    p：整数，默认为2。距离量度公式。当设置为2时使用欧氏距离公式，当设置为1时使用曼哈顿距离公式
# #    metric：默认为minkowski，用于距离度量，默认度量是minkowski，也就是p=2时的欧氏距离(欧几里得度量)
# #   metric_params ：距离公式的其它关键参数，这个直接默认即可
# #   n_jobs：并行处理设置。默认为1，临近点搜索并行工作数，如果为-1，那么cpu的cors都用于并行工作 
# #             
# clf = KNeighborsClassifier(n_neighbors=10,weights='uniform')
# clf.fit(X, Y) 
# #画出决策边界
# x_min,x_max = X[:,0].min() - 1,X[:,0].max() + 1
# y_min,y_max = X[:,1].min() - 1,X[:,1].max() + 1
# xx,yy = np.meshgrid(np.arange(x_min,x_max,0.02),  #meshgrid：用两个坐标轴上的点在平面上画网格
# np.arange(y_min,y_max,0.02)) 
# # reshape函数：在不更改数据的同时为数据赋予新形状：语法——np.reshape(a,newshape,order='C')
# #             参数一：需要传入的数组
# #             参数二：新形状应与原始形状兼容，如果是整数，则结果是该长度的一维数组，一个形状尺寸可以为-1。
# #                   在这种情况下，该值是根据数组的长度和纬度来判断的
# #             参数三：不知道 
# Z = clf.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)  
# plt.figure()   #绘图呗
# plt.pcolormesh(xx,yy,Z,cmap=cmap_light) 
# # #绘制预测结果图
# plt.scatter(X[:,0], X[:,1], c = Y,cmap = cmap_bold)
# plt.xlim(xx.min(),xx.max())
# plt.ylim(yy.min(),yy.max())
# plt.title('3_Class(k = 10,weights=uniform)')
# plt.show()


# # 支持向量机
# # 支持向量机是一种对线性和非线性进行分类的方法。SVM使用一种非线性映射，把原始数据映射到较高的
# # 的维度上，在新的维上，搜索最佳超平面。SVM分为三类：线性可分的线性SVM、线性不可分的线性SVM、
# # 非线性SVM。SVM可以用于数值预测和分类。对于数据非线性可分的情况，通过扩展线性SVM的方法，得到
# # 非线性的SVM，即采用非线性映射把输入数据变换到较高维空间，在新的空间搜索分离超平面
# # 算法：SVM的主要目标是找到最佳超平面，以便在不同类的数据点之间进行正确分类。超维度的平面等于
# # 输入特征的数量减去1.

# import numpy as np
# from sklearn import svm
# from sklearn import datasets
# from sklearn import metrics
# from sklearn import model_selection
# import matplotlib.pyplot as plt
# iris = datasets.load_iris()
# x,y = iris.data,iris.target
# #  model_selection.train_test_split的参数介绍：
# #      arrays：分割对象同样长度的列表或者numpm.arrays,矩阵
# #      test_size：两种指定方法：1.指定小数： 小数范围在0-1之间，它代表测试集所占的比例 
# #                            2.指定整数：整数的大小必须在这个数据集个数范围内 
# #      train_size：和test_size相似
# #      random_state:将分割的trainning和testing数据集打乱的个数设定，如果不指定的话，也可以
# #                   通过np.random来设定随机数
# #      shuffle和straify不常用，straify就是将数据分层
# x_train,x_test,y_train,y_test = model_selection.train_test_split(x,y,random_state=1,
#                                                                  test_size=0.2)
# classifier = svm.SVC(kernel='linear',gamma=0.1,decision_function_shape='ovo',C=0.1)
# classifier.fit(x_train,y_train.ravel())
# print('输出数据集的准确率为',classifier.score(x_train,y_train))
# print("输出测试集的准确率为",classifier.score(x_test,y_test))
# y_hat = classifier.predict(x_test)
# classreport = metrics.classification_report(y_test, y_hat)
# print(classreport)


# # 朴素贝叶斯1分类：贝叶斯分类是一类分类算法的总称,这类算法均以贝叶斯定理为基础，采用了概率推论
# # 方法。贝叶斯分类的原理是通过计算给定样本在各个类别上的后验概率，把该样本判定为最大后验概率
# # 所对应的类别，朴素贝叶斯分类是贝叶斯分类的一种，相对于贝叶斯分类，它假定所有的条件属性在
# # 类条件已知的情况下是完全相互独立的，这就极大概率地降低了条件概率计算的复杂度

# # 算法示例：
# # scilit-learn模块中有Naive Bayes子模块，包含了各种贝叶斯算法。利用贝叶斯分类器时首先
# # 设置分类器，然后利用训练样本进行训练和分类
# from sklearn.datasets import load_iris
# from sklearn.naive_bayes import GaussianNB
# iris = load_iris()
# clf = GaussianNB()   #设置分类器
# clf.fit(iris.data,iris.target)   #训练分类器
# y_pred = clf.predict(iris.data)   #预测
# print(iris.data.shape[0])
# print(iris.target != y_pred)

    
# # 聚类
# # 聚类是指将物理或抽象对象的集合分成由类似对象组成的多个子集的过程。每个子集是一个簇，使得
# # 簇中的对象彼此相似，但与其他簇中的对象尽量不同。聚类源于很多领域，包含数学，计算机科学，
# # 统计学等。常用的聚类方法有划分方法、层次方法和基于密度的方法等

# # 算法原理：给定一个包含n个对象或元组的数据库，使用一个划分方法构建数据的k个划分，每个划分
# #         表示一个簇，k<=n，而且满足：
# #           1.每个组至少包含一个对象
# #           2.每个对象属于仅属于一个组
# #         划分时要求同一个聚类中的对象尽可能地接近或相关，不同聚类中的对象尽可能地远离或不同
# #         一般来说，簇的表示方法有两种：
# #           1.k-均值表示法，由簇的平均值来代表整个簇
# #           2.k-中心点算法，由处于簇的中心区域的某个值代表整个簇 

# # k-means算法：
# # 输入：簇的数目K和包含n个对象的数据库
# # 输出：K个簇，使平方误差最小
# # 方法：
# #     1.随机选择K个对象，每个对象代表一个簇的初始1均值或中心
# #     2.对剩余的每个对象，根据它与簇均值的距离，将=它指派到最相似的簇
# #     3.计算每个簇的新均值
# #     4.回到步骤（2），循环，直到不再发生变化
# # 用于划分的k-means算法，其中每个簇的中心都用簇中所有对象的均值来表示

# # 算法示例：使用sklearn实现iris数据k-means聚类
# # 加载数据集
# from sklearn.datasets import load_iris
# from sklearn.cluster import KMeans
# iris = load_iris()
# # 构造k-means聚类模型
# X = iris.data
# estimator = KMeans(n_clusters = 3)
# # 对数据进行聚类
# estimator.fit(X)
# # 获取聚类标签
# label_pred = estimator.labels_
# # 显示各个样本所属的类别样本
# print(label_pred)

# 层次聚类
# 1.算法原理：层次聚类是按照某种方法进行层次分类，直到满足某种条件为止。层次聚类主要分为两类：
#           1.凝聚：从下到上。首先将每个对象作为一个簇，然后合并这些原子簇为越来越大的簇，直到
#             所有的对象都在一个簇中，或者满足某个终结条件
#           2.分裂：从上到下。首先将所有对象置于同一个簇中，然后逐渐细分为越来越小的簇，直到
#             每个对象自成一簇，或者达到了某个终止条件

# 2.层次聚类算法：输入：样本数据
#               输出：层次聚类结果
#               方法：1.将每个对象归为一类，共得到N类，每类仅包含一个对象。类和类之间的距离
#                      就是它们所包含的对象之间的距离
#                    2.找到最接近的两个类合并成一类，于是类的总数少了一个
#                    3.重新计算新的类与所有旧类之间的距离
#                    4.重复步骤2和步骤3，直到最后合并成一个类为止

# 层次聚类python实现：

 

 
     
     
     
     
     




