from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from graphviz import render

import os
import numpy as np
import matplotlib.pyplot as plt
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'


# Создаём решётку значений в виде матриц для каждой точки от min до max
def get_grid(data, eps: float = 0.01):
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    return np.meshgrid(np.arange(x_min, x_max, eps), np.arange(y_min, y_max, eps))


# Конвертируем dot в png
def dot_to_png(name: str):
    path = 'img/' + name + '.dot'
    render('dot', 'png', path)


# Проблемный случай для дерева и соседей. Когда легче разделить линейно.
# Строим массив в две колонки с равноудалёнными значениями
data, target = [], []
n = 500
x1_min = 0
x1_max = 30
x2_min = 0
x2_max = 30

for i in range(n):
    x1, x2 = np.random.randint(x1_min, x1_max), np.random.randint(x2_min, x2_max)

    if np.abs(x1 - x2) > 0.5:
        data.append([x1, x2])
        target.append(np.sign(x1 - x2))

X, y = np.array(data), np.array(target)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='autumn', edgecolors='black')
plt.show()

# Пытаемся разделить деревом
tree = DecisionTreeClassifier(random_state=17).fit(X, y)
xx, yy = get_grid(X, eps=.05)
predicted = tree.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.pcolormesh(xx, yy, predicted, cmap='autumn')
plt.scatter(X[:, 0], X[:, 1], c=y, s=100, cmap='autumn', edgecolors='black', linewidths=1.5)
plt.title('Easy task. Decision tree compexifies everything')

export_graphviz(tree, feature_names=['x1', 'x2'], out_file='img/deep_toy_tree.dot', filled=True)
dot_to_png('deep_toy_tree')

# Пытаемся разделить ближайшими соседями
knn = KNeighborsClassifier(n_neighbors=1).fit(X, y)
predicted = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.pcolormesh(xx, yy, predicted, cmap='autumn')
plt.scatter(X[:, 0], X[:, 1], c=y, s=100, cmap='autumn', edgecolors='black', linewidths=1.5)
plt.title('Easy task, kNN. Not bad')
plt.show()
