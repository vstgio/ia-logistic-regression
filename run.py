import pandas
import numpy as np
import matplotlib.pyplot as plt

values = pandas.read_csv("sample-data/ex2data1.csv", header=None).values

pos0 = np.where(values[:, 2] == 0)
pos1 = np.where(values[:, 2] == 1)

plt.scatter(values[pos0, 0], values[pos0, 1], marker='o', label="NOT ADMITTED")
plt.scatter(values[pos1, 0], values[pos1, 1], marker='+', label="ADMITTED")

plt.xlabel("EXAM 01 SCORE", labelpad=6, fontsize=8)
plt.ylabel("EXAM 02 SCORE",  labelpad=6, fontsize=8)
plt.legend(loc='lower right', bbox_to_anchor=(0, 1.02, 1, 0.2), ncol=2)

plt.show()
