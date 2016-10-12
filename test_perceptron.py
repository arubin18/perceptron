from perceptron import *

def binary_class (labels):
	""" convert strings to binary classifiers """
	for i in range(len(labels)):
		if labels[i] == "Iris-setosa":
			labels[i] = 0
		else:
			labels[i] = 1
	labels = labels.astype(float)
	return labels

a = genfromtxt("iris.txt", dtype=str, delimiter=',')[:100] 
data_test, data_train, labels_test, labels_train = partitioner_l(a, 40)
labels_test = binary_class(labels_test)
labels_train = binary_class(labels_train)
w, errors = perceptron(data_train, labels_train)

p_train = check_accuracy(data_train, labels_train, w)
p_test = check_accuracy(data_test, labels_test, w)

print ("TRAINING ACCURACY: " + str(p_train))
print ("TESTING ACCURACY: " + str(p_test))

show_line(data_train, labels_train)