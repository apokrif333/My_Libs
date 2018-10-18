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
При множестве количественных признаков, на каждом шаге будует прогоняться каждый признак по наибольшему приросту 
информации, и будет выбираться тот признак, который на данном разбиении системы обеспечит самый жадный прирост.
Случайный лес - строятся композиции дерерьев и устредняются ответы, подобно кросс-валидации.
Pruning (стржка) - строится дерево до максимальной глубины и потом, снизу-вверх, срезается, сравнивая разницу качества. 

DecisionTreeClassifier: 
max_depth – максимальная глубина дерева
max_features — максимальное число признаков, по которым ищется лучшее разбиение в дереве (проклятие масштабирования)
min_samples_leaf - указывает, при каком минимальном количестве элементов система будет дальше разделяться
'''


# Создаём решётку значений в виде матриц для каждой точки от min до max
def get_grid(data):
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    return np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))


# Конвертируем dot в png
def dot_to_png(name: str):
    path = 'C:/Users/Tom/PycharmProjects/Start/GibHub/My_Libs/img/' + name + '.dot'
    render('dot', 'png', path)


##DecisionTreeClassifier
# Применяем дерево решений на синтетических данных. Один Гауссов массив раскручивается вокруг 0, другой вокруг 2.
def test_des_tree_1():
    np.seed = 7
    train_data = np.random.normal(size=(100, 2))
    train_labels = np.zeros(100)
    train_data = np.r_[train_data, np.random.normal(size=(100, 2), loc=2)]
    train_labels = np.r_[train_labels, np.ones(100)]
    # plt.scatter(train_data[:, 0], train_data[:, 1], c=train_labels, s=50, cmap='autumn', edgecolors='black',
    #             linewidths=1.5)
    # plt.plot(range(-2, 5), range(4, -3, -1))

    # Параметры дерева.
    clf_tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=17)
    # Обучаем дерево
    clf_tree.fit(train_data, train_labels)
    # Отрисовыаем разделяющую поверхность (предсказание разделеления на той же самой выборке на которой обучалась)
    xx, yy = get_grid(train_data)
    predicted = clf_tree.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.pcolormesh(xx, yy, predicted, cmap='autumn')
    plt.scatter(train_data[:, 0], train_data[:, 1], c=train_labels, s=100, cmap='autumn', edgecolors='black',
                   linewidth=1.5)
    # Отображаем само дерево. Каждый класс имеет свой цвет в png дерева.
    export_graphviz(clf_tree, feature_names=['x1', 'x2'], out_file='my_first_DT.dot', filled=True)
    dot_to_png('my_first_DT')


# DecisionTreeClassifier
# Применяем дерево решений на синтетических данных. Создадим, таблицу с возрастом и невозвратом кредита.
def test_des_tree_2():
    data = pd.DataFrame({
        'Возраст': [17, 18, 20, 25, 29, 31, 33, 38, 49, 55, 64],
        'Зарплата': [25, 22, 36, 70, 33, 102, 88, 37, 59, 74, 80],
        'Невозрат кредита': [1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0]
    }).sort_values(by='Зарплата')
    age_tree = DecisionTreeClassifier(random_state=17)
    age_tree.fit(data['Возраст'].values.reshape(-1, 1), data['Невозрат кредита'].values)
    export_graphviz(age_tree, feature_names=['Возраст'], out_file='img/age_tree.dot', filled=True)
    dot_to_png('age_tree')

    age_sal_tree = DecisionTreeClassifier(random_state=17)
    age_sal_tree.fit(data[['Возраст', 'Зарплата']], data['Невозрат кредита'].values)
    export_graphviz(age_sal_tree, feature_names=['Возраст', 'Зарплата'], out_file='img/age_sal_tree.dot', filled=True)
    dot_to_png('age_sal_tree')


# DecisionTreeRegressor

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
    test_des_tree_2()
    plt.show()