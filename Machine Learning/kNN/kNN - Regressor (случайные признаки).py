from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

import numpy as np
import matplotlib.pyplot as plt

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

# Проблемный случай для соседа. Когда он не видит, что один из признаков пропорционален ответам
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
plt.show()

# И как финальный пример, обучим дерево
tree = DecisionTreeClassifier(random_state=17, max_depth=1)
tree_cv_score = np.mean(cross_val_score(tree, X_train, y_train, cv=5))
tree.fit(X_train, y_train)
tree_holdout_score = accuracy_score(y_holdout, tree.predict(X_holdout))
print(f'Decision tree. CV: {tree_cv_score}, holdout: {tree_holdout_score}')
