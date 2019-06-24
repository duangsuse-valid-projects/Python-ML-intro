#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Jun 24 13:38:24 2019

Iris dataset linear regression

@author: DuangSUSE
"""

from pandas import Series, read_csv

iris = read_csv('Iris.csv', encoding='utf-8', parse_dates=[], index_col=False)
iris['id'] = Series().astype(int)

###
def vectorize(w,i, cname='w', cid='id', iris=iris): iris.loc[iris[cname]== w, cid] = i
vectorize('setosa', 0); vectorize('versicolor', 1); vectorize('virginica', 2)

from sklearn.model_selection import train_test_split
iris_ds = iris.copy()

trainset, testset, trainsetid, testsetid = train_test_split(iris_ds, iris_ds['id'], train_size = 0.6)
del trainset['w'], trainset['id']

###
from sklearn.linear_model import LinearRegression
#from math import floor

lreg = LinearRegression()
lreg.fit(trainset, trainsetid)

testset_truthw, testset_truthid = testset['w'], testset['id']
del testset['w'], testset['id']
testset['predict'] = lreg.predict(testset)
testset['id'] = testset_truthid

##
from pandas import DataFrame
def verifyRegressionAccuracy(ts: DataFrame, emax: float = 0.1, npredict = 'predict', ntruth = 'real') -> float:
  predicteds, truths = ts[npredict], ts[ntruth]
  acceptables = [t for (r, t) in zip(truths, predicteds) if abs(t - r) <emax]
  return (len(acceptables) / len(predicteds)) *100

print("Error region 1.0: ", verifyRegressionAccuracy(testset, 1.0, 'predict', 'id'))

###: testset
vectorize('setosa', 1); vectorize('versicolor', 2); vectorize('virginica', 3)
iris_ds = iris.copy()

trainset, testset, trainsetid, testsetid = train_test_split(iris_ds, iris_ds['id'], train_size = 0.6)

del trainset['w'], trainset['id']

lreg = LinearRegression()
lreg.fit(trainset, trainsetid)

testset_truthw, testset_truthid = testset['w'], testset['id']
del testset['w'], testset['id']

testset['predict'] = lreg.predict(testset)
testset['id'] = testset_truthid

testset.describe()
print(testset.head)

###
testset['predict'] = testset['predict'].map(round)
print(verifyRegressionAccuracy(testset, 1, 'predict', 'id'))
