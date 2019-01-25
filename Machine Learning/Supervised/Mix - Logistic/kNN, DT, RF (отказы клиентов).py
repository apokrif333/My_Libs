from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from graphviz import render

import os
import pandas as pd
import numpy as np
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'


# Конвертируем dot в png
def dot_to_png(name: str):
    path = 'img/' + name + '.dot'
    render('dot', 'png', path)


# Сравниваем соседей, дерево и лес в классификации клиентов, которые уйдут от нас или нет
df = pd.read_csv('data/telecom_churn.csv')
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

# Проверем качество прогнозов на отложенной выборке
tree_pred = tree.predict(X_holdout)
print('DT on holdout: ', accuracy_score(y_holdout, tree_pred))
knn_pred = knn.predict(X_holdout)
print('kNN on holdout: ', accuracy_score(y_holdout, knn_pred))
print('RF cross_val score: ', np.mean(cross_val_score(forest, X_train, y_train, cv=5)))


# Проверим качество прогнозов дерева на кросс-валидации. GridSearchCV: для каждой уникальной пары значений
# параметров max_depth и max_features будет проведена 5-кратная кросс-валидация и выберется лучшее сочетание
# параметров.
tree_params = {'max_depth': range(1, 11),
               'max_features': range(4, 19)
               }
tree_grid = GridSearchCV(tree, tree_params, cv=5, n_jobs=4, verbose=True)
tree_grid.fit(X_train, y_train)

# Лучшее сочетание параметров и средняя доля правильных ответов кросс-валидации для дерева
print('DT best params for train: ', tree_grid.best_params_)
print('DT best score for train: ', tree_grid.best_score_)
print('DT accuracy for holdout: ', accuracy_score(y_holdout, tree_grid.predict(X_holdout)))

# Проверим качество прогнозов соседей на кросс-валидации.
knn_pipe = Pipeline([('scaler', StandardScaler()),
                     ('knn', KNeighborsClassifier(n_jobs=-1))
                     ])
knn_params = {'knn__n_neighbors': range(1, 10)}
knn_grid = GridSearchCV(knn_pipe, knn_params, cv=5, n_jobs=4, verbose=True)
knn_grid.fit(X_train, y_train)

# Лучшее сочетание параметров и средняя доля правильных ответов кросс-валидации для соседей
print('kNN best params for train: ', knn_grid.best_params_)
print('kNN best score for train: ', knn_grid.best_score_)
print('kNN accuracy for holdout: ', accuracy_score(y_holdout, knn_grid.predict(X_holdout)))

# Проверим качество прогнозов случайного леса на кросс-валидации.
forest_grid = GridSearchCV(forest, tree_params, cv=5, n_jobs=4, verbose=True)
forest_grid.fit(X_train, y_train)

# Лучшее сочетание параметров и средняя доля правильных ответов кросс-валидации для случайного леса
print('RF best params for train: ', forest_grid.best_params_)
print('RF best score for train: ', forest_grid.best_score_)
print('RF accuracy for holdout: ', accuracy_score(y_holdout, forest_grid.predict(X_holdout)))

# Лучшие результаты по качеству/ресурсам выдало дерево решений. Отрисуем его.
export_graphviz(tree_grid.best_estimator_, feature_names=df.columns, class_names=['Client', 'Leaver'],
                out_file='img/churn_tree.dot', filled=True)
dot_to_png('churn_tree')
