import pandas
import numpy as np
import matplotlib.pyplot as plt

def separate_values(values):
	x = values[:, :-1]
	y = values[:, -1]
	y = y.reshape(len(y), 1)
	x = np.hstack([np.ones((len(y), 1)), x])
	return x, y, np.where(values[:, 2] == 0), np.where(values[:, 2] == 1)

def sigmoid(value):
	return 1 / (1 + np.exp(-value))

def hypothesis(x, thetas):
	return sigmoid(np.dot(x, thetas))

def cost(x, y, thetas):
	return (1/len(y)) * np.sum((-y * np.log(hypothesis(x, thetas))) - (1 - y) * np.log(1 - hypothesis(x, thetas)))

def gradient():
	return 0

#------------------------#

values = pandas.read_csv("sample-data/ex2data1.csv", header=None).values
x, y, rejected, admitted = separate_values(values)
thetas = np.zeros((x.shape[1], 1))

plt.scatter(values[rejected, 0], values[rejected, 1], marker='.', label="NOT ADMITTED")
plt.scatter(values[admitted, 0], values[admitted, 1], marker='+', label="ADMITTED")

plt.xlabel("EXAM 01 SCORE", labelpad=6, fontsize=8)
plt.ylabel("EXAM 02 SCORE",  labelpad=6, fontsize=8)
plt.legend(loc='lower right', bbox_to_anchor=(0, 1.02, 1, 0.2), ncol=2)

#plt.show()
