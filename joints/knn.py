#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Example of kNN implemented from Scratch in Python
import csv
import random
import math
import operator
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold


class Distance:
    def euclideanDistance(self, instance1, instance2, length):
        distance = 0
        for x in range(length):
            distance += pow((instance1[x] - instance2[x]), 2)
        return math.sqrt(distance)

    def manhattanDistance(self, instance1, instance2, length):
        distance = 0
        for x in range(length):
            distance += abs(instance1[x] - instance2[x])
        return distance

    def cosineSimilarity(self, instance1, instance2, length):
        distance, sumv1, sumv2, sumv1v2 = 0, 0, 0, 0
        for i in range(length):
            x = instance1[i]
            y = instance2[i]
            sumv1 += x * x
            sumv2 += y * y
            sumv1v2 += x * y
        distance = sumv1v2 / (math.sqrt(sumv1) * math.sqrt(sumv2))
        return 1 - distance

    def minkowski_distance(self, p, q, n):
        return sum([abs(x-y)** n for x, y in zip(p[:-1], q[:-1])]) ** 1/n


#def loadDataset(filename, split, dataset=[], trainingIdx=[], testIdx=[], load=1):


    # Normalize
            
#def getDataset(dataset, trainingIdx, testIdx, trainingSet=[], testSet=[]):
#    for i in trainingIdx:
#        trainingSet.append(dataset[i])
#    for i in testIdx:
#        testSet.append(dataset[i])

# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])-2):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)-2):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


def getNeighbors(trainingSet, testInstance, k, mode=1,r = 1):
    distances = []
    length = len(testInstance)-1

    for x in range(len(trainingSet)):
        d = Distance()
        if mode == 1:
            dist = d.euclideanDistance(testInstance, trainingSet[x], length)
        elif mode == 2:
            dist = d.manhattanDistance(testInstance, trainingSet[x], length)
        elif mode == 3:
            dist = d.cosineSimilarity(testInstance, trainingSet[x], length)
        elif mode == 4:
            dist = d.minkowski_distance(testInstance, trainingSet[x], r)

        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1), reverse=False)
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def getPrediction(neighbors):
    mean = 0
    length = len(neighbors)
    for i in neighbors:
        i[-2] = float(i[-2])
        mean += i[-2]
    mean /= length
    return mean


def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(),
                         key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0


def mape(actual, predicted):
    prediction_error = 0
    for i in range(len(actual)):
        ac = float(actual[i][-2])
        print('Predicted : ' + repr(predicted[i]) + ' Actual : ' + repr(ac))
        if ac==0:
            prediction_error +=0
        else:
            prediction_error += abs((ac - predicted[i])/ac)
    return (abs(prediction_error/len(actual))) * 100


def main():
    # prepare data
    dataset = []
    trainingIdx = []
    testIdx = []
    # split = 0.67
    kfold = 10
    totalAccuracy = 0
    totalMape = 0
    r = 1
    print('1. Klasifikasi')
    print('2. Training data regresi')
    load = int(input('Pilih data (1-2) > '))
    # load = 2
    if load == 1:
        data = pd.read_csv('train.csv')
        data.drop(['id'], axis=1)
        dataset = data.values
        print(dataset)
        X = []
        Y = []
        # Convert String to Float
        for x in range(len(dataset)):
            for y in range(len(dataset[x])-1):
                dataset[x][y] = float(dataset[x][y])
    elif load == 2:
        loadDataset('new_data_training.csv', kfold,
                    dataset, trainingIdx, testIdx, load=2)
    else:
        print('Input salah')
        return
    k = int(input('Masukkan nilai k > '))
    # k = 3
    print('1. Euclidean Distance')
    print('2. Manhattan Distance')
    print('3. Cosine Similarity Distance')
    print('4. Minkowski Distance')
    inDist = int(input('Pilih distance (1-4) > '))
    # inDist = 4
    if inDist == 4:
        # r = 1
        r = int(input('Masukkan nilai r (r>0) > '))
        if r <= 0:
            print('Input salah')
            return

          # Calculate min and max for each column
    minmax = dataset_minmax(dataset)
    # Normalize columns
    normalize_dataset(dataset, minmax)
    # Split Classifier with others,X = Others, Y = Classifiers,
    for x in range(len(dataset)):
        X.append(dataset[x][:-1])
        Y.append(dataset[x][-1])
    # Get Idx of training and test set with StratifiedKFold
    if load == 1:
        kf = StratifiedKFold(n_splits=kfold)
        for train, test in kf.split(X, Y):
            trainingIdx.append(list(train))
            testIdx.append(list(test))
        print('trIdx = ' + repr(len(trainingIdx)) + ' testIdx = ' + repr(len(testIdx)))
    elif load == 2:
        kf = KFold(n_splits=split)
        for train, test in kf.split(dataset):
            trainingIdx.append(list(train))
            testIdx.append(list(test))
    for i in range(kfold):
        trainingSet = []
        testSet = []
        #getDataset(dataset, trainingIdx[i], testIdx[i], trainingSet, testSet)
        trainIdx = trainingIdx[i]
        testIdx = testIdx[i]
        for i in trainIdx:
            trainingSet.append(dataset[i])
        for i in testIdx:
            testSet.append(dataset[i])
        print('Train set: ' + str(len(trainingSet)))
        print('Test set: ' + str(len(testSet)))
        # print (dataset)
        # generate predictions
        predictions = []
        for x in range(len(testSet)):
            neighbors = getNeighbors(trainingSet, testSet[x], k, mode=inDist, r=r)
            if load == 1:
                result = getResponse(neighbors)
            elif load == 2:
                result = getPrediction(neighbors)
            predictions.append(result)
            # print('> predicted=' + str(result) + ', actual=' + str(testSet[x][-1]))
        if load == 1:
            accuracy = getAccuracy(testSet, predictions)
            totalAccuracy += accuracy
            print('Accuracy: ' + str(accuracy) + '%')
        elif load == 2:
            Mape = mape(testSet, predictions)
            totalMape += Mape
            print('MAPE: ' + str(Mape))
            print('\n')

    if load == 1:
        print('\nTotal Accuracy: ' + str(totalAccuracy/kfold) + '%')
    elif load == 2:
        mean = totalMape/kfold
        print('\nTotal Mean Absolute Percentage Error: ' + str(mean)+'%')
        print('Total Accuracy: ' + str(100-mean) + '%')


main()
