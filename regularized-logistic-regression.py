import pandas
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from sklearn.preprocessing import PolynomialFeatures

def separate_values(values):
	x = values[:, :-1]
	y = values[:, -1]
	y = y.astype(int)
	return x, y, np.where(values[:, 2] == 0), np.where(values[:, 2] == 1)

def sigmoid(value):
	return 1 / (1 + np.exp(-value))

def hypothesis(thetas, x):
	return sigmoid(np.dot(x, thetas))

def draw_points(values, rejected, admitted):
	plt.scatter(values[rejected, 0], values[rejected, 1], marker='.', label="APPROVED")
	plt.scatter(values[admitted, 0], values[admitted, 1], marker='+', label="REJECTED")
	plt.xlabel("MICROCHIP TEST 01", labelpad=6, fontsize=8)
	plt.ylabel("MICROCHIP TEST 02",  labelpad=6, fontsize=8)
	plt.legend(loc='lower right', bbox_to_anchor=(0, 1.02, 1, 0.2), ncol=2)

values = pandas.read_csv("sample-data/ex2data2.csv", header=None).values
x, y, rejected, admitted = separate_values(values)

p_features = PolynomialFeatures(6)
p_features = p_features.fit_transform(x)

draw_points(values, rejected, admitted)
plt.show()
