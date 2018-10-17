from __future__ import division, print_function
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from graphviz import render

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import warnings

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'


warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (10, 8)


''' t-SNE (t-distributed Stohastic Neighbor Embedding)
Найдем такое отображение из многомерного признакового пространства на плоскость (или в 3D, но почти всегда выбирают
2D), чтоб точки, которые были далеко друг от друга, на плоскости тоже оказались удаленными, а близкие точки – также
отобразились на близкие. То есть neighbor embedding – это своего рода поиск нового представления данных, при котором
сохраняется соседство.

Бинарные Yes/No-признаки переведем в числа (pd.factorize). Также нужно масштабировать выборку – из каждого признака
вычесть его среднее и поделить на стандартное отклонение, это делает StandardScaler.
'''


def t_SNE(df: pd.DataFrame, random: int, bool_column: pd.Series):
    X_scaled = StandardScaler().fit_transform(df)
    tsne_representation = TSNE(random_state=random).fit_transform(X_scaled)
    plt.scatter(tsne_representation[:, 0], tsne_representation[:, 1], c=bool_column.map({0: 'blue', 1: 'orange'}))


'''Деревья решений
Разбиваем данные по принципу жадного прироста информации (уменьшения энтропии). Указываем сколько может быть минимум
значений в системе после разбиения, указываем количество разбиений и т.д.
'''


# Создаём решётку значений в виде матриц для каждой точки от min до max
def get_grid(data):
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    return np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))


# Применяем дерево решений на синтетических данных. Один массив Гауссово раскручивается вокруг 0, другой вокруг 2.
def test_des_tree():
    np.seed = 7
    train_data = np.random.normal(size=(100, 2))
    train_labels = np.zeros(100)
    train_data = np.r_[train_data, np.random.normal(size=(100, 2), loc=2)]
    train_labels = np.r_[train_labels, np.ones(100)]
    # plt.scatter(train_data[:, 0], train_data[:, 1], c=train_labels, s=50, cmap='autumn', edgecolors='black',
    #             linewidths=1.5)
    # plt.plot(range(-2, 5), range(4, -3, -1))

    # Параметры дерева. Min_samples_leaf указывает, при каком минимальном количестве элементов будет дальше разделяться
    clf_tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=17)
    # Обучаем дерево
    clf_tree.fit(train_data, train_labels)
    # Отрисовыаем разделяющую поверхность (предсказание разделеления на той же самой выборке на которой обучалась)
    xx, yy = get_grid(train_data)
    predicted = clf_tree.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.pcolormesh(xx, yy, predicted, cmap='autumn')
    plt.scatter(train_data[:, 0], train_data[:, 1], c=train_labels, s=100, cmap='autumn', edgecolors='black',
                   linewidth=1.5)
    # Отображаем само дерево
    export_graphviz(clf_tree, feature_names=['x1', 'x2'], out_file='my_first_DT.dot', filled=True)
    path = 'C:/Users/Lex/PycharmProjects/Start/GitHub/My_Libs/my_first_DT.dot'
    render('dot', 'png', path)

# ----------------------------------------------------------------------------------------------------------------------
# Отрисовка Неопределённости Джини, энтропии, ошибки классификации.
def draw_entropy_and_Jini():
    xx = np.linspace(0, 1, 50)
    plt.plot(xx, [2 * x * (1-x) for x in xx], label='gini')
    plt.plot(xx, [4 * x * (1-x) for x in xx], label='2*gini')
    plt.plot(xx, [-x * np.log2(x) - (1-x) * np.log2(1-x) for x in xx], label='entropy')
    plt.plot(xx, [1 - max(x, 1-x) for x in xx], label='misscalss')
    plt.plot(xx, [2 - 2 * max(x, 1-x) for x in xx], label='2*missclass')
    plt.xlabel('p+')
    plt.ylabel('criterion')
    plt.title('Критерии качества как функции от p+ (бинарная классификация)')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    test_des_tree()
    plt.show()