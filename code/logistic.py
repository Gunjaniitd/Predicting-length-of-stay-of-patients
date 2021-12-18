import sys
import os

import math
import time

import numpy as pylib
import pandas as pdlib
from scipy.special import softmax
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectKBest, chi2

start = 0.0
prev = 0.0

def OHE(trainfile, testfile):

	trainData = pdlib.read_csv(trainfile, index_col = 0)  
	testData = pdlib.read_csv(testfile, index_col = 0) 

	Y = pylib.array(trainData["Length of Stay"])
	Y = pdlib.get_dummies(Y).astype('float').to_numpy()

	trainData = trainData.drop(columns = ["Length of Stay"])

	n, m = trainData.shape

	allData = pdlib.concat([trainData, testData], ignore_index = True)
	allData = pdlib.get_dummies(allData, columns = trainData.columns[:-1], drop_first=True)
	allData = allData.to_numpy()

	X = allData[ : n , :].astype('float')
	test = allData[ n : , : ].astype('float')

	X = pylib.concatenate(( pylib.ones((X.shape[0], 1)), X), axis = 1)
	test = pylib.concatenate(( pylib.ones((test.shape[0], 1)), test), axis = 1)

	return X, Y, test

def lossFunction(W, X, Y):
	e = 1e-15
	prediction = softmax((X @ W).T, axis = 0).T
	return -1 * pylib.sum(pylib.log(pylib.clip(pylib.sum(Y * prediction, axis = 1), e, 1-e)))/X.shape[0]

def updateLR(method, learning_info, itr, X, Y, W):
	if (method == 1):
		return float(learning_info)

	if (method == 2):
		return float(learning_info) / (math.sqrt(itr))

	if (method == 3):
		n, a, b = list( map( float, learning_info.split(',') ) )
		
		loss = lossFunction(W, X, Y)

		result = softmax((X @ W).T, axis = 0).T
		gradient = ( (X.T @ (result - Y) ) / X.shape[0] )

		while ( ( lossFunction(W - n * gradient, X, Y) - loss ) > (-1 * a * n * (pylib.linalg.norm(gradient) ** 2 )) ):
			n = b * n

		return n

def updateWeights(X, Y, weights, learning_rate):
	n, m = X.shape
	result = softmax((X @ weights).T, axis = 0).T
	gradient = ( (X.T @ (result - Y) ) / n )
	return (weights - (learning_rate * gradient))

def saveFiles(weights, test, weightfile, outputfile):
	pylib.savetxt(weightfile, weights, delimiter = "\n")
	probabilities =  softmax((test @ weights).T, axis = 0).T
	prediction = pylib.argmax(probabilities, axis = 1) + 1
	pylib.savetxt(outputfile, prediction, delimiter = "\n")

def logisticRegression(part, trainfile, testfile, paramfile, outputfile, weightfile):
	global start
	global prev

	if (part == "a"):

		X, Y, test = OHE(trainfile, testfile)

		file = open(paramfile, "r")
		lines = file.readlines()
		file.close()

		method, learning_info, iterCount = int(lines[0]), lines[1], int(lines[2])

		n, m = X.shape
		k = Y.shape[1]
		weights = pylib.zeros((m, k))

		for itr in range(1, iterCount + 1):
			learning_rate = updateLR(method, learning_info, itr, X, Y, weights)

			# print(itr)
			# print(learning_rate)

			weights = updateWeights(X, Y, weights, learning_rate)

		saveFiles(weights, test, weightfile, outputfile)

	elif (part == "b"):

		X, Y, test = OHE(trainfile, testfile)

		file = open(paramfile, "r")
		lines = file.readlines()
		file.close()

		method, learning_info, iterCount, batchSize = int(lines[0]), lines[1], int(lines[2]), int(lines[3])
		
		n, m = X.shape
		k = Y.shape[1]
		weights = pylib.zeros((m, k))

		loopCount = n // batchSize

		for itr in range(1, iterCount + 1):
			learning_rate = updateLR(method, learning_info, itr, X, Y, weights)

			# print(itr)
			# print(learning_rate)

			for count in range(loopCount):
				batch_x = X[count * batchSize: batchSize * (1 + count), :]
				batch_y = Y[count * batchSize: batchSize * (1 + count), :]

				weights = updateWeights(batch_x, batch_y, weights, learning_rate)

		saveFiles(weights, test, weightfile, outputfile)

	elif (part == "c"):

		X, Y, test = OHE(trainfile, testfile)

		method, learning_info, iterCount, batchSize = 2, 10, 500, 100
		
		n, m = X.shape
		k = Y.shape[1]
		weights = pylib.zeros((m, k))

		loopCount = n // batchSize

		for itr in range(1, iterCount + 1):
			learning_rate = learning_info / (math.sqrt(itr + 12))

			# print(itr)
			# print(learning_rate)
			# print(lossFunction(weights, X, Y))

			for count in range(loopCount):
				batch_x = X[count * batchSize: batchSize * (1 + count), :]
				batch_y = Y[count * batchSize: batchSize * (1 + count), :]

				weights = updateWeights(batch_x, batch_y, weights, learning_rate)


				timeTaken = time.time() - start
				print(timeTaken)
				if ( (timeTaken > prev + 60) and (timeTaken < 590) ):
					prev = timeTaken
					saveFiles(weights, test, weightfile, outputfile)

		timeTaken = time.time() - start
		if (timeTaken < 590):
			saveFiles(weights, test, weightfile, outputfile)

	elif (part == "d"):

		X, Y, test = OHE(trainfile, testfile)

		model = SelectKBest(chi2, k = 500)
		X = model.fit_transform(X, Y)
		test = model.transform(test)

		method, learning_info, iterCount, batchSize = 2, 10, 250, 100
		
		n, m = X.shape
		k = Y.shape[1]
		weights = pylib.zeros((m, k))

		loopCount = n // batchSize

		for itr in range(1, iterCount + 1):
			learning_rate = learning_info / (math.sqrt(itr + 12))

			# print(itr)
			# print(learning_rate)
			# print(lossFunction(weights, X, Y))

			for count in range(loopCount):
				batch_x = X[count * batchSize: batchSize * (1 + count), :]
				batch_y = Y[count * batchSize: batchSize * (1 + count), :]

				weights = updateWeights(batch_x, batch_y, weights, learning_rate)


				timeTaken = time.time() - start
				if ( (timeTaken > prev + 60) and (timeTaken < 890) ):
					prev = timeTaken
					saveFiles(weights, test, weightfile, outputfile)

		timeTaken = time.time() - start
		if (timeTaken < 890):
			saveFiles(weights, test, weightfile, outputfile)


if __name__ == '__main__':
	start = time.time()
	prev = 0.0

	part = sys.argv[1]

	if (part == "a"):

		train = sys.argv[2]
		test = sys.argv[3]
		param = sys.argv[4]
		outputfile = sys.argv[5]
		weightfile = sys.argv[6]

		logisticRegression("a", train, test, param, outputfile, weightfile)

	elif (part == "b"):

		train = sys.argv[2]
		test = sys.argv[3]
		param = sys.argv[4]
		outputfile = sys.argv[5]
		weightfile = sys.argv[6]

		logisticRegression("b", train, test, param, outputfile, weightfile)


	elif (part == "c"):

		train = sys.argv[2]
		test = sys.argv[3]
		outputfile = sys.argv[4]
		weightfile = sys.argv[5]

		logisticRegression("c", train, test, "param.txt", outputfile, weightfile)

	elif (part == "d"):

		train = sys.argv[2]
		test = sys.argv[3]
		outputfile = sys.argv[4]
		weightfile = sys.argv[5]

		logisticRegression("d", train, test, "param.txt", outputfile, weightfile)


