# file: partition.py
# ------------------
# Partitions data into two arrays

from numpy import *
import numpy as np
import random

def partitioner (data_train, test_examples):
	""" Partitions the training and test sets of the data
	by randomly assigning a test_examples percentage of the data points
	to the test set and removing those points from the 
	original data. The training set takes on the remaining
	data points from the orginal data. """

	m, n = data_train.shape
	data_test = ones((1, n))
	num_partitions = int(float(test_examples)/100 * m)

	for i in range(num_partitions):
		index = random.randint(0, m-1)
		# Append data points to test set
		data_test = vstack((data_test, data_train[index]))
		# Remove test set data points from training set
		data_train = np.delete(data_train, index, 0)
		m -= 1

	data_test = data_test[1:]

	return data_test, data_train

def partitioner_l (data, test_examples):
	""" Partitions the training and test sets of the data with labels
	by randomly assigning a test_examples percentage of the data points
	to the test set and removing those points from the 
	original data. The training set takes on the remaining
	data points from the orginal data. """

	m, n = data.shape
	labels_train = data[:, n-1]

	data_train = data[:, :n-1]
	data_train = data_train.astype(float)
	m, n = data_train.shape

	data_test = ones((1, n))
	num_partitions = int(float(test_examples)/100 * m)
	labels_test = array([1]) # 1 x 1 matrix

	for i in range(num_partitions):
		index = random.randint(0, m-1)
		# Append data points to test set
		data_test = vstack((data_test, data_train[index]))
		labels_test = vstack((labels_test, labels_train[index]))
		# Remove test set data points from training set
		data_train = np.delete(data_train, index, 0)
		labels_train = np.delete(labels_train, index, 0)
		m -= 1

	data_test = data_test[1:]
	labels_test = labels_test[1:]

	return data_test, data_train, labels_test, labels_train
