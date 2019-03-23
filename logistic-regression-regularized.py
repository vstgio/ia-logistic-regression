import pandas
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from sklearn.preprocessing import PolynomialFeatures

def separate_values(values):
    x = values[:, :-1]
    y = values[:, -1]
    y = y.reshape(len(y), 1)
    y = y.astype(int)
    return x, y, np.where(values[:, 2] == 0), np.where(values[:, 2] == 1)

def sigmoid(value):
	return 1 / (1 + np.exp(-value))

def hypothesis(thetas, x):
	return sigmoid(np.dot(x, thetas))

def cost(thetas, x, y, lambda_value):
    cost = (1/len(y)) * np.sum((-y * np.log(hypothesis(thetas, x))) - (1 - y) * np.log(1 - hypothesis(thetas, x)))
    regularization = (lambda_value/(2*len(y))) * (np.dot(thetas[1:].T, thetas[1:]))
    return (cost + regularization)

def gradient(thetas, x, y, lambda_value):
    grad = (1/len(y)) * (np.dot(x.T, (hypothesis(thetas, x) - y)))
    grad[1:] = grad[1:] + (lambda_value / len(y)) * thetas[1:]
    return grad

def accuracy(x, y, opt_thetas, boundary_value):
	results = hypothesis(opt_thetas, x)
	results = (results > boundary_value).astype(int)
	return np.mean(results == y) * 100

def draw_points(values, rejected, admitted):
	plt.scatter(values[rejected, 0], values[rejected, 1], marker='.', label="APPROVED")
	plt.scatter(values[admitted, 0], values[admitted, 1], marker='+', label="REJECTED")
	plt.xlabel("MICROCHIP TEST 01", labelpad=6, fontsize=8)
	plt.ylabel("MICROCHIP TEST 02",  labelpad=6, fontsize=8)
	plt.legend(loc='lower right', bbox_to_anchor=(0, 1.02, 1, 0.2), ncol=2)

def draw_decision_boundary(opt_thetas, x, y, poly):
    x1_min, x1_max = np.min(x[:, 1]), np.max(x[:, 1])
    x2_min, x2_max = np.min(x[:, 2]), np.max(x[:, 2])
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))

    h = sigmoid(poly.fit_transform(np.c_[xx1.ravel(), xx2.ravel()]).dot(opt_thetas))
    h = h.reshape(xx1.shape)
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='g')

#------------------------#

values = pandas.read_csv("sample-data/ex2data2.csv", header=None).values
x, y, rejected, admitted = separate_values(values)

poly = PolynomialFeatures(6)
p_features = poly.fit_transform(x)
initial_thetas = np.zeros((p_features.shape[1], 1))

opt_thetas = scipy.optimize.fmin_tnc(func=cost, x0=initial_thetas.flatten(), fprime=gradient, args=(p_features, y.flatten(), 1))
opt_thetas = opt_thetas[0].reshape(opt_thetas[0].shape[0],1)

print("")
print("::: Accuracy of the model: {0:.2f}%".format(accuracy(p_features, y, opt_thetas, 0.5)))
print("")

draw_points(values, rejected, admitted)
draw_decision_boundary(opt_thetas, p_features, y, poly)
plt.show()
