#!/bin/python
# -*- coding: utf-8 -*-
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.cross_validation import train_test_split

'''
train_test_split / fit 不需要用到pandas. 最基礎numpy array即可.
pandas用在資料讀入csv/寫出csv.

'''

# 讀入鳶尾花資料
iris = load_iris()

iris_X = iris.data # 2D numpy.ndarray (array of 1D vector), len = 150
iris_y = iris.target # 1D numpy.ndarray of int labels, len = 150 

# 切分訓練與測試資料. 
train_X, test_X, train_y, test_y = train_test_split(iris_X, iris_y, test_size = 0.3)

#len(test_X)=len(test_Y)=45 (150 * 0.3 ) len(train_X)=len(train_Y)=105

# 建立分類器
clf = tree.DecisionTreeClassifier()
iris_clf = clf.fit(train_X, train_y)

# 預測
test_y_predicted = iris_clf.predict(test_X)
print(test_y_predicted)

# 標準答案
print(test_y)