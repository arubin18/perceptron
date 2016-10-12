# file: perceptron.py
# -------------------
# Classification algorithm

from numpy import *
import random
import matplotlib.pyplot as plt
import numpy as np
from partitioner import *

def unit_step(x):
	if x < 0:
		return 0
	else:
		return 1

def perceptron (data, labels):
	""" returns modified weights and errors.
	Parameters: data is the learning data without labels,
	labels are the binary classifiers """

	m,n = data.shape

	w = np.random.rand(n)
	errors = []
	eta = 0.2 # learning rate
	count = 1000 # number of learning iterations

	for i in range(count):
		d = i % len(data)
		result = dot(w, data[d])
		expected = labels[d]
		error = expected - unit_step(result)
		errors.append(error)
		w += eta*error*data[d]

	return w, errors

def show_line(data, labels):
	plt.figure()
	for i in range(0, len(data)):
		if labels[i] == 0.:
			plt.scatter(data[i,0], data[i,1], marker = ">")
		else:
			plt.scatter(data[i,0], data[i,1], marker = "+")
	plt.show()

def check_accuracy (data, labels, weights):
	""" Use weights to classify data points and check the accuracy """
	count = 0
	gs = []
	rs = []
	for x in range(0,len(data)):
		results = dot(data[x], weights)
		guess = unit_step(results)
		gs.append(guess) # append prediction
		rs.append(labels[x]) # append result
		if guess - labels[x] == 0:
			count += 1

	percentage = ((float(count) / len(data)) * 100)

	return percentage
