from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(seed=42)

X = np.zeros((150, 2))
X[:50, 0] = np.random.normal(loc=0.0, scale=.3, size=50)
X[:50, 1] = np.random.normal(loc=0.0, scale=.3, size=50)
X[50:100, 0] = np.random.normal(loc=2.0, scale=.5, size=50)
X[50:100, 1] = np.random.normal(loc=-1.0, scale=.2, size=50)
X[100:150, 0] = np.random.normal(loc=1.0, scale=.2, size=50)
X[100:150, 1] = np.random.normal(loc=2.0, scale=.5, size=50)
plt.scatter(x=X[:, 0], y=X[:, 1])
plt.show()

# Считаем верхний треугольник матрицы попарных расстояний
distance_mat = pdist(X)
Z = hierarchy.linkage(distance_mat, 'single')  # агломеративный алго
print(X)
print('-------')
print(distance_mat)
print('-------')
print(Z)

plt.figure(figsize=(10, 5))
dn = hierarchy.dendrogram(Z, color_threshold=0.5)
plt.show()
