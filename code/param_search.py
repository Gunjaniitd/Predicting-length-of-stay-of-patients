import sys
import os

import math

import numpy as pylib
import pandas as pdlib
from scipy.special import softmax

import time

def OHE(trainData, testData, Y):
	n, m = trainData.shape

	allData = pdlib.concat([trainData, testData], ignore_index = True)
	allData = pdlib.get_dummies(allData, columns = trainData.columns[:-1], drop_first=True)
	allData = allData.to_numpy()

	train = allData[ : n , :].astype('float')
	test = allData[ n : , : ].astype('float')

	train = pylib.concatenate(( pylib.ones((train.shape[0], 1)), train), axis = 1)
	test = pylib.concatenate(( pylib.ones((test.shape[0], 1)), test), axis = 1)

	result = pdlib.get_dummies(Y).astype('float').to_numpy()

	return train, result, test

def lossFunction(W, X, Y):
	e = 1e-15
	prediction = softmax((X @ W).T, axis = 0).T
	return -1 * pylib.sum(pylib.log(pylib.clip(pylib.sum(Y * prediction, axis = 1), e, 1-e)))/X.shape[0]

def UpdateLR(method, eta, alpha, beta, itr, X, Y, W):
	if (method == 1):
		return eta

	if (method == 2):
		return eta / (math.sqrt(itr))

	if (method == 3):
		n, a, b = [eta, alpha, beta]
		
		loss = lossFunction(W, X, Y)

		result = softmax((X @ W).T, axis = 0).T
		gradient = ( (X.T @ (result - Y) ) / X.shape[0] )

		while ( ( lossFunction(W - n * gradient, X, Y) - loss ) > (-1 * a * n * (pylib.linalg.norm(gradient) ** 2 )) ):
			n = b * n

		return n

def logisticRegression(X, Y, test, method, eta, alpha, beta, batchSize):

	iterCount = 250
	
	n, m = X.shape
	k = Y.shape[1]
	weights = pylib.zeros((m, k))

	loopCount = n // batchSize

	file = open("method" + str(method) + "size" + str(batchSize) + ".csv", "w")

	start = time.time()

	for itr in range(1, iterCount + 1):
		learning_rate = UpdateLR(method, eta, alpha, beta, itr, X, Y, weights)

		# print(itr)
		# print(learning_rate)

		if (itr % 10 == 1):
			loss = lossFunction(weights, X, Y)
			end = time.time()
			timeTaken = end - start
			file.write(str(loss) + " , " + str(timeTaken) + "\n")

		for count in range(loopCount):
			batch_x = X[count * batchSize: batchSize * (1 + count), :]
			batch_y = Y[count * batchSize: batchSize * (1 + count), :]
			
			result = softmax((batch_x @ weights).T, axis = 0).T
			gradient = ( (batch_x.T @ (result - batch_y) ) / batchSize )
			weights = weights - (learning_rate * gradient)

	file.close() 

def paramSearch(trainfile, testfile):

	trainData = pdlib.read_csv(trainfile , index_col = 0)   
	Y = pylib.array(trainData["Length of Stay"])
	trainData = trainData.drop(columns = ["Length of Stay"])
	testData = pdlib.read_csv(testfile , index_col = 0)
	X, Y, test = OHE(trainData, testData, Y)

	batches = [50, 100, 200, 400, 800, 1600]

	# print("Working on Method 1")
	for batchSize in batches:
		logisticRegression(X, Y, test, 1, 0.02, 0.4, 0.5, batchSize)

	# print("Working on Method 2")
	for batchSize in batches:
		logisticRegression(X, Y, test, 2, 5, 0.4, 0.5, batchSize)

	# print("Working on Method 3")
	for batchSize in batches:
		logisticRegression(X, Y, test, 3, 2.5, 0.4, 0.5, batchSize)

	# etas = [50, 25]

	# for eta in etas:
	# 	logisticRegression(X, Y, test, 2, eta, 0.4, 0.5, 100)

	# alphas = [0.1, 0.2, 0.3, 0.4, 0.5]

	# for alpha in alphas:
	# 	logisticRegression(X, Y, test, 3, 2.5, alpha, 0.5, 100)

	# betas = [0.1, 0.3, 0.5, 0.7, 0.9]

	# for beta in betas:
	# 	logisticRegression(X, Y, test, 3, 2.5, 0.4, beta, 100)

if __name__ == '__main__':
	train = sys.argv[1]
	test = sys.argv[2]

	paramSearch(train, test)







