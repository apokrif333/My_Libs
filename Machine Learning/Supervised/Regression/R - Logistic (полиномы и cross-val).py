from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Логистическая регрессия
data = pd.read_csv('data/microchip_tests.txt', header=None, names=('test1', 'test2', 'released'))
# Сохраним признаки и целевой класс в Numpy.
X = data.iloc[:, :2].values
y = data.ix[:, 2].values

plt.scatter(X[y == 1, 0], X[y == 1, 1], c='green', label='Выпущен')
plt.scatter(X[y == 0, 0], X[y == 0, 1], c='red', label='Бракован')
plt.xlabel('Тест 1')
plt.ylabel('Тест 2')
plt.title('2 теста микрочипов')
plt.legend()
plt.show()

# Добавляем полиноминальные признаки и обучим лог. регрессию с заданой регуляризацией
poly = PolynomialFeatures(degree=7)
X_poly = poly.fit_transform(X)

C = 10_000  # 1e-2, 1, 10_000
logit = LogisticRegression(C=C, n_jobs=4, random_state=17).fit(X_poly, y)

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
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='green', label='Выпущен')
plt.scatter(X[y == 0, 0], X[y == 0, 1], c='red', label='Бракован')
plt.xlabel('Тест 1')
plt.ylabel('Тест 2')
plt.title(f'2 теста микрочипов. Логит с С={C}')
plt.legend()
plt.show()

print('Доля правильных ответов классификатора на обучающей выборке: ', round(logit.score(X_poly, y), 3))

# Найдём оптимальное значение С (регулиризации)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
c_values = np.logspace(-1, 3, 500)
logit_searcher = LogisticRegressionCV(Cs=c_values, cv=skf, verbose=1, n_jobs=4).fit(X_poly, y)
avg_iteration_score = logit_searcher.scores_[1].mean(0)

# Дальнейший код весьма прост. Можно сравнить качество LogisticRegression, просто выведя на один график logit.score
# при С от 0 до 1000, а на другой LogisticRegressionCV выведя все значения logit.score при С от -2 до 3.
score_list = []
for C in np.linspace(0.1, 1000, 500):
    logit = LogisticRegression(C=C, n_jobs=4, random_state=17).fit(X_poly, y)
    score_list.append(round(logit.score(X_poly, y), 3))

plt.plot(c_values, avg_iteration_score, label='Cross_Values')
plt.plot(np.linspace(0.01, 1000, 500), score_list, label='Direction Teaching')
plt.legend()
plt.show()
