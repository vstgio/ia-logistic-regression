import pandas
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

def separate_values(values):
	x = values[:, :-1]
	y = values[:, -1]
	y = y.reshape(len(y), 1)
	y = y.astype(int)
	x = np.hstack([np.ones((len(y), 1)), x])
	return x, y, np.where(values[:, 2] == 0), np.where(values[:, 2] == 1)

def sigmoid(value):
	return 1 / (1 + np.exp(-value))

def hypothesis(thetas, x):
	return sigmoid(np.dot(x, thetas))

def cost(thetas, x, y):
	return (1/len(y)) * np.sum((-y * np.log(hypothesis(thetas, x))) - (1 - y) * np.log(1 - hypothesis(thetas, x)))

def gradient(thetas, x, y):
	return (1/len(y)) * np.dot(x.T, (hypothesis(thetas, x)-y))

def draw_points(values, rejected, admitted):
	plt.scatter(values[rejected, 0], values[rejected, 1], marker='.', label="NOT ADMITTED")
	plt.scatter(values[admitted, 0], values[admitted, 1], marker='+', label="ADMITTED")
	plt.xlabel("EXAM 01 SCORE", labelpad=6, fontsize=8)
	plt.ylabel("EXAM 02 SCORE",  labelpad=6, fontsize=8)
	plt.legend(loc='lower right', bbox_to_anchor=(0, 1.02, 1, 0.2), ncol=2)

def draw_decision_line(best_thetas, x, y):
	x1_min, x1_max = np.min(x[:, 1]), np.max(x[:, 1])
	x2_min, x2_max = np.min(x[:, 2]), np.max(x[:, 2])
	xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))

	h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0], 1)), xx1.ravel(), xx2.ravel()].dot(best_thetas))
	h = h.reshape(xx1.shape)

	plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')

def accuracy(x, y, best_thetas, boundary_value):
	results = hypothesis(best_thetas.reshape(3,1), x)
	results = (results > boundary_value).astype(int)
	return np.mean(results == y) * 100

#------------------------#

values = pandas.read_csv("sample-data/ex2data1.csv", header=None).values
x, y, rejected, admitted = separate_values(values)
initial_thetas = np.zeros((x.shape[1], 1))

best_thetas = scipy.optimize.fmin_tnc(func=cost, x0=initial_thetas, fprime=gradient, args=(x, y.flatten()))

draw_points(values, rejected, admitted)
draw_decision_line(best_thetas[0], x, y)

print("")
print("::: Accuracy of the model: {}%".format(accuracy(x, y, best_thetas[0], 0.5)))
print("")

plt.show()
