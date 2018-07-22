#!/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.cross_validation import train_test_split

'''
    from list data to np array
'''

def by_list():
    input_x = [
            [5. , 3.6, 1.4, 0.2],
            [5.4, 3.9, 1.7, 0.4],
            [4.6, 3.4, 1.4, 0.3],
            [5. , 3.4, 1.5, 0.2],
            [4.4, 2.9, 1.4, 0.2],
            [4.4, 2.9, 1.4, 0.2], # 故意弄成一樣, 用np.unique排除
        ]

    input_y = [0, 0, 0, 0, 0, 0]

    iris_X = np.array(input_x)
    iris_y = np.array(input_y)
    print "np unique=", np.unique(iris_X, axis=0) # 合併完全一模一樣的samples;要加axis=0, 否則會變成1D array,裡面是所有不同的數值集合(0.2,0.3,0.4...)
    print "iris_X=", iris_X, type(iris_X)
    print "iris_y=", iris_y, type(iris_y)

    #iris_X = iris.data # 2D numpy.ndarray (array of 1D vector), len = 150
    #iris_y = iris.target # 1D numpy.ndarray of int labels, len = 150 

    # 切分訓練與測試資料. 
    train_X, test_X, train_y, test_y = train_test_split(iris_X, iris_y, test_size = 0.3)

    #len(test_X)=len(test_Y)=45 (150 * 0.3 ) len(train_X)=len(train_Y)=105
    # 

    # 建立分類器
    clf = tree.DecisionTreeClassifier()
    iris_clf = clf.fit(train_X, train_y)

    # 預測
    test_y_predicted = iris_clf.predict(test_X)
    print(test_y_predicted)

    # 標準答案
    print(test_y)

def by_stackup():
    temp_x = [
            [0, 5. , 3.6, 1.4, 0.2],
            [1, 5.4, 3.9, 1.7, 0.4],
            [2, 4.6, 3.4, 1.4, 0.3],
            [3, 5. , 3.4, 1.5, 0.2],
            [4, 4.4, 2.9, 1.4, 0.2],
            [4, 4.4, 2.9, 1.4, 0.2], # 故意弄成一樣, 用np.unique排除
        ]

    tdata = np.unique(temp_x, axis=0)

    x = np.vstack((i[1],i[2],i[3],i[4]) for i in tdata)
    y = tdata[:, 0]

    print "X=", x, type(x)
    print "Y=", y, type(y)

    # 切分訓練與測試資料. 
    train_X, test_X, train_y, test_y = train_test_split(x, y, test_size = 0.3)

    # 建立分類器
    clf = tree.DecisionTreeClassifier()
    iris_clf = clf.fit(train_X, train_y)

    # 預測
    test_y_predicted = iris_clf.predict(test_X)
    print(test_y_predicted)

    # 標準答案
    print(test_y)

#by_list()
by_stackup()