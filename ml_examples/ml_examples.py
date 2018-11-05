from __future__ import division, print_function
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.datasets import load_files
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score, validation_curve, learning_curve
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline, make_pipeline
from graphviz import render

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (10, 8)


# Создаём решётку значений в виде матриц для каждой точки от min до max
def get_grid(data, eps: float=0.01):
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    return np.meshgrid(np.arange(x_min, x_max, eps), np.arange(y_min, y_max, eps))


# Конвертируем dot в png
def dot_to_png(name: str):
    path = 'img/' + name + '.dot'
    render('dot', 'png', path)


# Применяем к каждому из множества значений формулу
def func_for_elements(x: list):
    x = x.ravel()
    return np.exp(-x ** 2) + 1.5 * np.exp(-(x - 2) ** 2)


# Генерируем рандом-семплы с некоторым шумом
def generate_samples_and_noise(n_samples, noise):
    X = np.random.rand(n_samples) * 10 - 5
    X = np.sort(X).ravel()
    y = np.exp(-X ** 2) + 1.5 * np.exp(-(X - 2) ** 2) + np.random.normal(0.0, noise, n_samples)
    X = X.reshape((n_samples, 1))
    return X, y


# Выдаём данные на график о средней ошибке, и её отклонении
def plot_with_err(x, data, **kwargs):
    mu, std = data.mean(1), data.std(1)
    lines = plt.plot(x, mu, '-', **kwargs)
    plt.fill_between(x, mu - std, mu + std, edgecolor='none', facecolor=lines[0].get_color(), alpha=0.2)


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


# DecisionTreeClassifier
# Применяем дерево решений на синтетических данных. Один Гауссов массив раскручивается вокруг 0, другой вокруг 2.
def test_des_tree_1():
    np.seed = 7
    train_data = np.random.normal(size=(100, 2))
    train_labels = np.zeros(100)
    train_data = np.r_[train_data, np.random.normal(size=(100, 2), loc=2)]
    train_labels = np.r_[train_labels, np.ones(100)]
    plt.scatter(train_data[:, 0], train_data[:, 1], c=train_labels, s=50, cmap='autumn', edgecolors='black',
                linewidths=1.5)
    plt.plot(range(-2, 5), range(4, -3, -1))

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
# Применяем дерево решений на синтетических данных. Тут, для решения количественной классификации.
def test_des_tree_3():
    n_train = 150
    n_test = 1000
    noise = 0.1

    X_train, y_train = generate_samples_and_noise(n_samples=n_train, noise=noise)
    X_test, y_test = generate_samples_and_noise(n_samples=n_test, noise=noise)

    reg_tree = DecisionTreeRegressor(max_depth=5, random_state=17)
    reg_tree.fit(X_train, y_train)
    reg_tree_pred = reg_tree.predict(X_test)

    plt.plot(X_test, func_for_elements(X_test), "b")
    plt.scatter(X_train, y_train, c='b', s=20)
    plt.plot(X_test, reg_tree_pred, 'g', lw=2)
    plt.xlim([-5, 5])
    plt.title('Decision tree regressor, MSE = %.2f' % np.sum((y_test - reg_tree_pred) ** 2))


'''Метод ближайших соседей
k Nearest Neighbors, или kNN - посмотри на соседей, какие преобладают, таков и ты. Формально основой метода является 
гипотеза компактности: если метрика расстояния между примерами введена достаточно удачно, то схожие примеры гораздо чаще 
лежат в одном классе, чем в разных.
sklearn.neighbors.KNeighborsClassifier:
    - weights: "uniform" (все веса равны), "distance" (вес обратно пропорционален расстоянию до тестового примера) или 
    другая определенная пользователем функция
    - algorithm (опционально): "brute", "ball_tree", "KD_tree", или "auto". В первом случае ближайшие соседи для каждого 
    тестового примера считаются перебором обучающей выборки. Во втором и третьем — расстояние между примерами хранятся в 
    дереве, что ускоряет нахождение ближайших соседей. В случае указания параметра "auto" подходящий способ нахождения 
    соседей будет выбран автоматически на основе обучающей выборки.
    - leaf_size (опционально): порог переключения на полный перебор в случае выбора BallTree или KDTree для нахождения 
    соседей
    - metric: "minkowski", "manhattan", "euclidean", "chebyshev" и другие
'''


def nearest_neighbors_and_tree_and_random_forest():
    df = pd.read_csv('test_data/telecom_churn.csv')
    df['International plan'] = pd.factorize(df['International plan'])[0]
    df['Voice mail plan'] = pd.factorize(df['Voice mail plan'])[0]
    df['Churn'] = df['Churn'].astype('int')
    states = df['State']
    y = df['Churn']
    df.drop(['State', 'Churn'], axis=1, inplace=True)

    # Разбиваем данные для отложенной выборки и обучаем
    X_train, X_holdout, y_train, y_holdout = train_test_split(df.values, y, test_size=0.3, random_state=17)
    tree = DecisionTreeClassifier(max_depth=5, random_state=17)
    knn = KNeighborsClassifier(n_neighbors=10)
    forest = RandomForestClassifier(max_depth=5, random_state=17)
    tree.fit(X_train, y_train)
    knn.fit(X_train, y_train)
    forest.fit(X_train, y_train)

    ''' # Проверем качество прогнозов на отложенной выборке 
    tree_pred = tree.predict(X_holdout)
    print(accuracy_score(y_holdout, tree_pred))
    knn_pred = knn.predict(X_holdout)
    print(accuracy_score(y_holdout, knn_pred))
    print(np.mean(cross_val_score(forest, X_train, y_train, cv=5)))
    '''

    # Проверим качество прогнозов дерева на кросс-валидации. GridSearchCV: для каждой уникальной пары значений
    # параметров max_depth и max_features будет проведена 5-кратная кросс-валидация и выберется лучшее сочетание
    # параметров.
    tree_params = {
        'max_depth': range(1, 11),
        'max_features': range(4, 19)
    }
    tree_grid = GridSearchCV(tree, tree_params, cv=5, n_jobs=-1, verbose=True)
    tree_grid.fit(X_train, y_train)

    # Лучшее сочетание параметров и средняя доля правильных ответов кросс-валидации для дерева
    print(tree_grid.best_params_)
    print(tree_grid.best_score_)
    print(accuracy_score(y_holdout, tree_grid.predict(X_holdout)))

    # Проверим качество прогнозов соседей на кросс-валидации.
    knn_pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_jobs=-1))])
    knn_params = {
        'knn__n_neighbors': range(1, 10)
    }
    knn_grid = GridSearchCV(knn_pipe, knn_params, cv=5, n_jobs=-1, verbose=True)
    knn_grid.fit(X_train, y_train)

    # Лучшее сочетание параметров и средняя доля правильных ответов кросс-валидации для соседей
    print(knn_grid.best_params_)
    print(knn_grid.best_score_)
    print(accuracy_score(y_holdout, knn_grid.predict(X_holdout)))

    # Проверим качество прогнозов случайного леса на кросс-валидации.
    forest_grid = GridSearchCV(forest, tree_params, cv=5, n_jobs=-1, verbose=True)
    forest_grid.fit(X_train, y_train)

    # Лучшее сочетание параметров и средняя доля правильных ответов кросс-валидации для случайного леса
    print(forest_grid.best_params_)
    print(forest_grid.best_score_)
    print(accuracy_score(y_holdout, forest_grid.predict(X_holdout)))

    # Лучшие результаты по качеству/ресурсам выдало дерево решений. Отрисуем его.
    export_graphviz(tree_grid.best_estimator_, feature_names=df.columns, class_names=['Client', 'Leaver'],
                    out_file='img/churn_tree.dot', filled=True)
    dot_to_png('churn_tree')


# Проблемный случай для соседа. Когда он не видит, что один из признаков пропорционален ответам
def neighbors_problem():
    n_obj = 1_000
    np.seed = 17
    y = np.random.choice([-1, 1], size=n_obj)
    # Первый признак пропорционален целевому
    x1 = 0.3 * y
    # Создаём 99 колонок с 1000 случайных значений
    x_other = np.random.random(size=[n_obj, 100 - 1])
    # Присоединяем x1 колонку к рандом колонкам
    X = np.hstack([x1.reshape([n_obj, 1]), x_other])

    # Построим кривые, отражающие эффективность на кросс и отложенной, при различных значениях соседей
    X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.3, random_state=17)
    cv_scores, holdout_scores = [], []
    n_neighb = [1, 2, 3, 5] + list(range(50, 550, 50))
    for k in n_neighb:
        knn = KNeighborsClassifier(n_neighbors=k)
        cv_scores.append(np.mean(cross_val_score(knn, X_train, y_train, cv=5)))
        knn.fit(X_train, y_train)
        holdout_scores.append(accuracy_score(y_holdout, knn.predict(X_holdout)))

    plt.plot(n_neighb, cv_scores, label='CV')
    plt.plot(n_neighb, holdout_scores, label='holdout')
    plt.title('Easy task. kNN fails')
    plt.legend()

    # И как финальный пример, обучим дерево
    tree = DecisionTreeClassifier(random_state=17, max_depth=1)
    tree_cv_score = np.mean(cross_val_score(tree, X_train, y_train, cv=5))
    tree.fit(X_train, y_train)
    tree_holdout_score = accuracy_score(y_holdout, tree.predict(X_holdout))
    print(f'Decision tree. CV: {tree_cv_score}, holdout: {tree_holdout_score}')


''' Случайный лес
class sklearn.ensemble.RandomForestRegressor

n_estimators - число деревьев в лесу (по дефолту - 10)
criterion - функция, которая измеряет качество разбиения ветки (mse и mae)
max_features - число признаков, по которым ищется разбиение. Число, процент или выбрать из auto (все признаки), 'sqrt', 
'log2'
max_depth - максимальная глубина дерева
min_samples_split - минимальное количество объектов, необходимых для разделения внутри узла. Число или процент от общего
числа объектов
min_samples_leaf - минимальное число объектов в листе. Число или процент от общего числа объектов
min_weight_fraction_leaf - мин. взвешенная доля от общей суммы весов (всех входных объектов). По дефолту одинаковый вес
max_leaf_nodes - максимальное количество листьев. По дефолту нет ограничения.
min_impurity_split - порог для остановки наращивания дерева. По дефолту 1e-7
bootstrap - применять ли бутстрэп для построения дерева. По дефолту True
oob_score - использовать ли out_of_bag объекты для оценки R^2. По дефолту False
n_jobs - количество ядер для построения. -1 использовать все ядра. 
random_state - начальное значение для генерации случаных чисел. По дефолту не установлено.
verbose - вывод логов по построению. 
warm_start - использует уже натренированную модель и добавляет в ансамбль.

class sklearn.ensemble.RandomForestClassifier

criterion - gini или entropy
class_weight - вес каждого класса. По дефолту все 1. (можно передать словарь с весами или balanced, тогда веса классов 
будут равны их исходным частям в генеральной совокупности. balanced_subsample, тогда веса на каждой подвыборке будут 
меняться в зависимости от распределения классов на этой подвыборке).
'''


'''Регрессия
--------------------------
'''


# Логистическая регрессия
def logictic_regression():
    data = pd.read_csv('test_data/microchip_tests.txt', header=None, names=('test1', 'test2', 'released'))

    # Сохраним признаки с целевой класс в Numpy.
    X = data.iloc[:, :2].values
    y = data.ix[:, 2].values

    # plt.scatter(X[y ==1, 0], X[y == 1, 1], c='green', label='Выпущен')
    # plt.scatter(X[y == 0, 0], X[y == 0, 1], c='red', label='Бракован')
    # plt.xlabel('Тест 1')
    # plt.ylabel('Тест 2')
    # plt.title('2 теста микрочипов')
    # plt.legend()

    # Добавляем полиноминальные признаки и обучим лог. регрессию с заданой регуляризацией
    poly = PolynomialFeatures(degree=7)
    X_poly = poly.fit_transform(X)

    C = 10_000  # 1e-2, 1, 10_000
    logit = LogisticRegression(C=C, n_jobs=-1, random_state=17).fit(X_poly, y)

    # Код для отображения разделяющей кривой классификатора
    grid_step = .01
    poly_featurizer = poly

    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_step), np.arange(y_min, y_max, grid_step))
    # Ставим цвета в каждой точке
    Z = logit.predict(poly_featurizer.transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, cmap=plt.cm.Paired)

    # plt.scatter(X[y == 1, 0], X[y == 1, 1], c='green', label='Выпущен')
    # plt.scatter(X[y == 0, 0], X[y == 0, 1], c='red', label='Бракован')
    # plt.xlabel('Тест 1')
    # plt.ylabel('Тест 2')
    # plt.title(f'2 теста микрочипов. Логит с С={C}')
    # plt.legend()

    print('Доля правильных ответов классификатора на обучающей выборке: ', round(logit.score(X_poly, y), 3))

    # Найдём оптимальное значение С (регулиризации)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
    c_values = np.logspace(-2, 3, 500)
    logit_searcher = LogisticRegressionCV(Cs=c_values, cv=skf, verbose=1, n_jobs=-1).fit(X_poly, y)
    # Дальнейший код весьма прост. Можно сравнить качество LogisticRegression, просто выведя на один график logit.score
    # при С от 0 до 1000, а на другой LogisticRegressionCV выведя все значения logit.score при С от -2 до 3.

# Различные примеры-----------------------------------------------------------------------------------------------------
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


# Проблемный случай для дерева и соседей. Когда легче разделить линейно.
def tree_and_neighbors_problem():
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
    # plt.scatter(X[:, 0], X[:, 1], c=y, cmap='autumn', edgecolors='black')

    # Пытаемся разделить деревом
    tree = DecisionTreeClassifier(random_state=17).fit(X, y)
    xx, yy = get_grid(X, eps=.05)
    predicted = tree.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    # plt.pcolormesh(xx, yy, predicted, cmap='autumn')
    # plt.scatter(X[:, 0], X[:, 1], c=y, s=100, cmap='autumn', edgecolors='black', linewidths=1.5)
    # plt.title('Easy task. Decision tree compexifies everything')

    export_graphviz(tree, feature_names=['x1', 'x2'], out_file='img/deep_toy_tree.dot', filled=True)
    dot_to_png('deep_toy_tree')

    # Пытаемся разделить ближайшими соседями
    knn = KNeighborsClassifier(n_neighbors=1).fit(X, y)
    predicted = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.pcolormesh(xx, yy, predicted, cmap='autumn')
    plt.scatter(X[:, 0], X[:, 1], c=y, s=100, cmap='autumn', edgecolors='black', linewidths=1.5)
    plt.title('Easy task, kNN. Not bad')


# Обучим сеть разпознавать рукописные цифры деревом и соседями
def numbers_reader():
    # Загружаем дату, где цифры в виде матриц 8х8, каждое значение элемента - интенсивность белого
    from sklearn.datasets import load_digits
    data = load_digits()
    X, y = data.data, data.target
    X[0, :].reshape([8, 8])

    # Создадим в plt 4 суб-плота и в каждом выведем 4 первые цифры из даты
    f, axes = plt.subplots(1, 4, sharey=True, figsize=(16, 6))
    for i in range(4):
        axes[i].imshow(X[i, :].reshape([8, 8]))

    # Создаём отложенную выборку, обучаем сети и прогнозируем выборку
    X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.3, random_state=17)
    tree = DecisionTreeClassifier(max_depth=5, random_state=17).fit(X_train, y_train)
    knn = KNeighborsClassifier(n_neighbors=10).fit(X_train, y_train)
    tree_pred = tree.predict(X_holdout)
    knn_pred = knn.predict(X_holdout)
    print(accuracy_score(y_holdout, tree_pred), accuracy_score(y_holdout, knn_pred))

    # Создадим кросс-валидацию для дерева. Теперь признаков 64, так как цветовая модель из 8х8
    tree_params = {
        'max_depth': [1, 2, 3, 5, 10, 20, 25, 30, 40, 50, 64],
        'max_features': [1, 2, 3, 5, 10, 20, 30, 50, 64]
    }
    tree_grid = GridSearchCV(tree, tree_params, cv=5, n_jobs=-1, verbose=True).fit(X_train, y_train)
    print(tree_grid.best_params_, tree_grid.best_score_)

    # Один сосед и случайный лес
    print(np.mean(cross_val_score(KNeighborsClassifier(n_neighbors=1), X_train, y_train, cv=5)))
    print(np.mean(cross_val_score(RandomForestClassifier(random_state=17), X_train, y_train, cv=5)))


# Определяем хороший отзыв о фильме или нет, при помощи логистической регресиии
def movie_review_logictic_reg():
    path = 'test_data/Del/aclImdb'
    reviews_train = load_files(os.path.join(path, 'train'), categories=['pos', 'neg'])
    text_train, y_train = reviews_train.data, reviews_train.target
    reviews_test = load_files(os.path.join(path, 'test'), categories=['pos', 'neg'])
    text_test, y_test = reviews_test.data, reviews_test.target
    print(f"Number of documents in training data: {len(text_train)}\n",
          f"Number of documents in test data: {len(text_test)}\n",
          np.bincount(y_test))

    # Создадим словарь всех слов с индексом
    cv = CountVectorizer().fit(text_train)
    print(len(cv.vocabulary_))
    print(cv.get_feature_names()[:50])
    print(cv.get_feature_names()[50_000:50_050])

    X_train = cv.transform(text_train)
    X_test = cv.transform(text_test)

    # Обучим логист. регрессию
    logit = LogisticRegression(n_jobs=-1, random_state=7).fit(X_train, y_train)
    print('logit without C on train and test' + str(round(logit.score(X_train, y_train), 3)),
                                                    str(round(logit.score(X_test, y_test), 3)))

    # Тут должна быть фукнция, но вынесем её пока в тело. Отрисосываем веса ключевых слов после обучения
    classifier = logit
    feature_names = cv.get_feature_names()
    n_top_features = 25
    coef = classifier.coef_.ravel()
    positive_coefficients = np.argsort(coef)[-n_top_features:]
    negative_coefficients = np.argsort(coef)[:n_top_features]
    interesting_coefficients = np.hstack([negative_coefficients, positive_coefficients])

    plt.figure(figsize=(15 ,5))
    colors = ['red' if c < 0 else 'blue' for c in coef[interesting_coefficients]]
    plt.bar(np.arange(2 * n_top_features), coef[interesting_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(1, 1 + 2 * n_top_features), feature_names[interesting_coefficients], rotation=60, ha='right')
    plt.show()

    # Подбираем коэфф. регулиризации для лог. регрессии. sklearn.pipeline говорит сначала применить CountVectorizer, а
    # затем обучить регрессию. Так мы избегаем подсматривания в тестовую выборку.
    text_pipe_logit = make_pipeline(CountVectorizer(), LogisticRegression(n_jobs=-1, random_state=7)).fit(text_train,
                                                                                                          y_train)
    print('HO on test by train ' + str(text_pipe_logit.score(text_test, y_test)))

    param_grid_logit = {'logisticregression__C': np.logspace(-5, 0, 6)}
    grid_logit = GridSearchCV(text_pipe_logit, param_grid_logit, cv=3, n_jobs=-1).fit(text_train, y_train)
    print('CV on train with C: ' + str(grid_logit.best_params_), str(grid_logit.best_score_))

    grid = grid_logit
    param_name = 'logisticregression__C'
    plt.plot(grid.param_grid[param_name], grid.cv_results_['mean_train_score'], color='green', label='train')
    plt.plot(grid.param_grid[param_name], grid.cv_results_['mean_test_score'], color='red', label='test')
    plt.legend()

    print('From CV on train to test: ' + str(grid_logit.score(text_test, y_test)))

    # Сравним лог. регрессию и случ. лес
    forest = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=17).fit(X_train, y_train)
    print('Fores from train to test: ' + str(round(forest.score(X_test, y_test), 3)))


# XOR проблема, т.е. исключающее ИЛИ, как проблема для линейных моделей
def xor_problem():
    # Создаем дата-сет с исключающием или. Или одно, или другое.
    rng = np.random.RandomState(0)  # Это подобие random_seed, только сохраняется на будущее, без вызова
    X = rng.randn(200, 2)
    print(X)
    y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
    plt.scatter(X[:, 0], X[:, 1], s=30, c=y, cmap=plt.cm.Paired)

    # Попытаемся обучить линейной
    clf = LogisticRegression()
    plot_title = 'Logictic Regression, XOR problem'

    xx, yy = np.meshgrid(np.linspace(-3, 3, 50), np.linspace(-3, 3, 50))
    clf.fit(X, y)
    Z = clf.predict_proba(np.vstack((xx.ravel(), yy.ravel())).T)[:, 1]  # Прогнозируем вероятности
    Z = Z.reshape(xx.shape)

    image = plt.imshow(Z, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
                       origin='lower', cmap=plt.cm.PuOr_r)
    contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2, linetypes='--')
    plt.scatter(X[:, 0], X[:, 1], s=30, c=y, cmap=plt.cm.Paired)
    plt.xticks(())
    plt.yticks(())
    plt.xlabel(r'$<!--math>$inline$x_1$inline$</math -->$')
    plt.ylabel(r'$<!--math>$inline$x_2$inline$</math -->$')
    plt.axis([-3, 3, -3, 3])
    plt.colorbar(image)
    plt.title(plot_title, fontsize=12)
    plt.show()


    # Подадим полиноминальные признаки, т.е. создадим 6-мерное пространство
    logit_pipe = Pipeline([('poly', PolynomialFeatures(degree=2)), ('logit', LogisticRegression())])
    clf = logit_pipe
    plot_title = 'Logictic Regression + quadratic features. XOR problem'

    xx, yy = np.meshgrid(np.linspace(-3, 3, 50), np.linspace(-3, 3, 50))
    clf.fit(X, y)
    Z = clf.predict_proba(np.vstack((xx.ravel(), yy.ravel())).T)[:, 1]  # Прогнозируем вероятности
    Z = Z.reshape(xx.shape)

    image = plt.imshow(Z, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
                       origin='lower', cmap=plt.cm.PuOr_r)
    contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2, linetypes='--')
    plt.scatter(X[:, 0], X[:, 1], s=30, c=y, cmap=plt.cm.Paired)
    plt.xticks(())
    plt.yticks(())
    plt.xlabel(r'$<!--math>$inline$x_1$inline$</math -->$')
    plt.ylabel(r'$<!--math>$inline$x_2$inline$</math -->$')
    plt.axis([-3, 3, -3, 3])
    plt.colorbar(image)
    plt.title(plot_title, fontsize=12)
    plt.show()


# Попытка понять, что лучше Простить/Усложнить модель, Добавить больше признаков, Накопить больше данных
def main_DS_problem():
    data = pd.read_csv('test_data/telecom_churn.csv').drop('State', axis=1)
    data['International plan'] = data['International plan'].map({'Yes': 1, 'No': 0})
    data['Voice mail plan'] = data['Voice mail plan'].map({'Yes': 1, 'No': 0})
    y = data['Churn'].astype('int').values
    X = data.drop('Churn', axis=1).values

    # Используем стохастический град. спуск и построим валид. кривые по обучающей и проверочной выборке
    alphas = np.logspace(-2, 0, 20)  # коэффициеты регуляризации
    sgd_logit = SGDClassifier(loss='log', n_jobs=-1, random_state=17)
    logit_pipe = Pipeline(
        [('scaler', StandardScaler()), ('poly', PolynomialFeatures(degree=2)), ('sgd_logit', sgd_logit)])
    val_train, val_test = validation_curve(logit_pipe, X, y, 'sgd_logit__alpha', alphas, cv=5, scoring='roc_auc')

    # Обученные данные выводим на график. Тренировочная.
    x = alphas
    data = val_train
    plot_with_err(alphas, val_train, label='training scores')

    # Обученные данные выводим на график. Тестировочная.
    data = val_test
    plot_with_err(alphas, val_test, label='validation scores')

    plt.xlabel(r'$\alpha$'); plt.ylabel('ROC AUC')
    plt.legend()
    plt.show()

    # Результат нас не удовлетворяет. Построим кривые обучения, чтобы понять, нужно ли нам накопить больше данных
    degree = 2
    alpha = 10  # коэфф. регуляризации. Для примера взято значение 10.
    train_sizes = np.linspace(0.05, 1, 20)
    logit_pipe = Pipeline([('scalae', StandardScaler()), ('poly', PolynomialFeatures(degree=degree)),
                           ('sgd_logit', SGDClassifier(n_jobs=-1, random_state=17, alpha=alpha))])
    N_train, val_train, val_test = learning_curve(logit_pipe, X, y, train_sizes=train_sizes, cv=5, scoring='roc_auc')
    plot_with_err(N_train, val_train, label='trainings scores')
    plot_with_err(N_train, val_test, label='validation scores')
    plt.xlabel('Training Set Size'); plt.ylabel('AUC')
    plt.legend()
    plt.show()

    # Как видно из кривых обучения, для небольшого объёма данных ошибки сильно отличаются, так как на небольшом объёме
    # быстро возникает переобучение.
    # А для больших объёмов, ошибки сходятся, что указывает на недообучение.
    # Вывод: если добавить ещё данных, ошибка на обучающей не будет расти, но и на тестовых не будет уменьшаться. То
    # есть ошибки сошлись и добавление новых данных не поможет.
    # Значит надо улучшать модель. Исходя из первого графика видим, что мы можем взять коэфф. регуляризации 0.05
    # Тогда мы видим, что кривые сходятся и ешё не стали параллельными, а значит в этом случае новые данные окажут
    # эффект, чтобы повысить качество валидации.
    # Если усложнить модель и выставить коэфф. регуляризации на 10 ** -4, то возникнет переобучение, качество модели
    # постоянно падает, как на обучении, так и на валидации.


if __name__ == '__main__':
    main_DS_problem()
    plt.show()