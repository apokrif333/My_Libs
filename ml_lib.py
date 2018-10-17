from __future__ import division, print_function
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (10, 6)


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


# Применяем дерево решений на синтетических данных
def test_des_tree():
    np.seed = 7
    train_data = np.random.normal(size=(100, 2))


draw_entropy_and_Jini()
