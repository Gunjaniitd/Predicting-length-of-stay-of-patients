import sys
import os

import math

import numpy as pylib
import pandas as pdlib
from scipy.special import softmax
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectKBest, chi2

import time

def lossFunction(W, X, Y):
	e = 1e-15
	prediction = softmax((X @ W).T, axis = 0).T
	return -1 * pylib.sum(pylib.log(pylib.clip(pylib.sum(Y * prediction, axis = 1), e, 1-e)))/X.shape[0]

def logisticRegression(X, Y, actualY, method, eta, batchSize):

	iterCount = 250
	
	n, m = X.shape
	k = Y.shape[1]
	weights = pylib.zeros((m, k))

	loopCount = n // batchSize

	start = time.time()

	for itr in range(1, iterCount + 1):
		learning_rate = eta / (math.sqrt(itr + 12))

		# print(itr)
		# print(learning_rate)

		for count in range(loopCount):
			batch_x = X[count * batchSize: batchSize * (1 + count), :]
			batch_y = Y[count * batchSize: batchSize * (1 + count), :]
			
			result = softmax((batch_x @ weights).T, axis = 0).T
			gradient = ( (batch_x.T @ (result - batch_y) ) / batchSize )
			weights = weights - (learning_rate * gradient)


		# print(itr)
		# if (itr % 10 == 0):
		# 	probabilities =  softmax((X @ weights).T, axis = 0).T
		# 	prediction = pylib.argmax(probabilities, axis = 1) + 1

		# 	score = f1_score(actualY, prediction, average = 'micro')

		# 	print(score)


def featureSelection(trainfile, testfile):
	trainData = pdlib.read_csv(trainfile , index_col = 0)   

	impFeature = ["Facility Name", "Hospital County", "APR Risk of Mortality"]

	s = set()

	for i in range(len(impFeature)):
		for j in range(i + 1, len(impFeature)):
			trainData[impFeature[i] + " " + impFeature[j]] = 500 * trainData[impFeature[i]] + trainData[impFeature[j]]
			s.add(impFeature[i] + " " + impFeature[j])

	toRemove = ["Operating Certificate Number", "Facility Name", "Age Group", "Gender", "Birth Weight", "Payment Typology 1", "Payment Typology 2", "Payment Typology 3"]
	toRemove = []
	train = trainData.drop(columns = toRemove)

	for column in train.columns:
		if column not in s:
			continue

		gp = (train.groupby(column)["Length of Stay"]).agg(["count", "mean"])
		fre = gp["count"]
		mean = gp["mean"]
		train[column] = train[column].map(mean)

	Y = pylib.array(trainData["Length of Stay"])

	train = train.drop(columns = ["Length of Stay"])
	train = pdlib.get_dummies(train, columns = train.columns[:-7], drop_first=True)
	train = train.to_numpy()
	train = train.astype('float')
	train = pylib.concatenate(( pylib.ones((train.shape[0], 1)), train), axis = 1)

	model = SelectKBest(chi2, k = 500)
	X_new = model.fit_transform(train, Y)
	# print(X_new.shape)

	actualY = Y
	Y = pdlib.get_dummies(Y).astype('float').to_numpy()

	logisticRegression(X_new, Y, actualY, 2, 10, 100)

if __name__ == '__main__':
	train = sys.argv[1]
	test = sys.argv[2]

	featureSelection(train, test)	






