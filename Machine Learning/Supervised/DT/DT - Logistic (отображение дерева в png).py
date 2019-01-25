from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
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
    path = name + '.dot'
    render('dot', 'png', path)


'''Деревья решений
Разбиваем данные по принципу жадного прироста информации (уменьшения энтропии). Указываем сколько может быть минимум
значений в системе после разбиения, указываем количество разбиений и т.д.
При множестве количественных признаков, на каждом шаге будует прогоняться каждый признак по наибольшему приросту 
информации, и будет выбираться тот признак, который на данном разбиении системы обеспечит самый жадный прирост.
Случайный лес - строятся композиции дерерьев и устредняются ответы, подобно кросс-валидации.
Pruning (стржка) - строится дерево до максимальной глубины и потом, снизу-вверх, срезается, сравнивая разницу качества. 

DecisionTreeClassifier: 
max_depth – максимальная глубина дерева
max_features — максимальное число признаков, по которым ищется лучшее разбиение в дереве (проклятие масштабирования)
min_samples_leaf - указывает, при каком минимальном количестве элементов система будет дальше разделяться
'''

# DecisionTreeClassifier
# Применяем дерево решений на синтетических данных. Один Гауссов массив раскручивается вокруг 0, другой вокруг 2.
np.seed = 7
train_data = np.random.normal(size=(100, 2))
train_labels = np.zeros(100)
train_data = np.r_[train_data, np.random.normal(size=(100, 2), loc=2)]
train_labels = np.r_[train_labels, np.ones(100)]
plt.scatter(train_data[:, 0], train_data[:, 1], c=train_labels, s=50, cmap='autumn', edgecolors='black', linewidths=1.5)
plt.plot(range(-2, 5), range(4, -3, -1))

# Параметры дерева.
clf_tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=17)
clf_tree.fit(train_data, train_labels)
# Отрисовыаем разделяющую поверхность (предсказание разделеления на той же самой выборке на которой обучалась)
xx, yy = get_grid(train_data)
predicted = clf_tree.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.pcolormesh(xx, yy, predicted, cmap='autumn')
plt.scatter(train_data[:, 0], train_data[:, 1], c=train_labels, s=100, cmap='autumn', edgecolors='black', linewidth=1.5)
# Отображаем само дерево. Каждый класс имеет свой цвет в png дерева.
export_graphviz(clf_tree, feature_names=['x1', 'x2'], out_file='my_first_DT.dot', filled=True)
dot_to_png('my_first_DT')
