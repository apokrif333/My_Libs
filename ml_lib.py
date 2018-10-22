from __future__ import division, print_function
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from graphviz import render

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings


# Создание обученного дерева (классификатора или регрессора или случаного леса)
def des_tree(criterion: str, depth: int, random: int, X_train, y_train, tree: str):
    if tree == 'class':
        return DecisionTreeClassifier(criterion=criterion, max_depth=depth, random_state=random).fit(X_train, y_train)
    elif tree == 'regress':
        return DecisionTreeRegressor(criterion=criterion, max_depth=depth, random_state=random).fit(X_train, y_train)
    elif tree == 'forest':
        return RandomForestClassifier(criterion=criterion, max_depth=depth, random_state=random).fit(X_train, y_train)
    else:
        print(f'Передано неверное значение tree для выбора дерева: {tree}')


# Обучаем дерево по кросс-валидации
def cross_valid_tree(des_tree_params, depth: list, features: list, cv_samples: int, X_train, y_train):
    return GridSearchCV(des_tree_params, max_depth=depth, max_features=features, cv=cv_samples, n_jobs=-1,
                        verbose=True).fit(X_train, y_train)


# Классифицируем данные методом ближайших соседей
def kNN_create(nighbors: int, X_train, y_train):
    return KNeighborsClassifier(n_neighbors=nighbors).fit(X_train, y_train)


# Обучаем соседей по кросс-валидации
def cross_valid_kNN(neighbors: list, cv_samples: int, X_train, y_train):
    knn_pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_jobs=-1))])
    return GridSearchCV(knn_pipe, knn__n_neighbors=neighbors, cv=cv_samples, n_jobs=-1, verbose=True).fit(X_train,
                                                                                                          y_train)

# Предсказываем данные по обученной модели
def tree_predict(model, X_holdout):
    return model.predict(X_holdout)


# Создание отложенной выборки X_train, X_holdout, y_train, y_holdout
def hold_out_create(df: pd.DataFrame, y: pd.Series, test: float, random: int):
     return train_test_split(df.values, y, test_size=test, random_state=random)


# Оценка кросс-валидации по средней c созданием кросс-валидации
def cv_mean(trained_model, X_train, y_train, cv_samples: int):
    return np.mean(cross_val_score(trained_model, X_train, y_train, cv=cv_samples))


# Выводим аккураси
def print_accuracy(y_holdout, X_holdout):
    print(accuracy_score(y_holdout, X_holdout))


# Выводим качество кросс-валидации
def cv_quality(model_grid, y_holdout, X_holdout):
    print(model_grid.best_params_)
    print(model_grid.best_score_)
    print(accuracy_score(y_holdout, model_grid.predict(X_holdout)))


# Создаём dot-file c обученным деревом
def create_dot(clf_tree, feature_names: list, file_name: str ):
    export_graphviz(clf_tree, feature_names=feature_names, out_file=file_name + '.dot', filled=True)


# Конвертируем dot в png
def dot_to_png(name: str):
    path = 'img/' + name + '.dot'
    render('dot', 'png', path)


# Конвертация значений столбца в бинарные признаки
def encoder():
    preprocessing.LabelEncoder()
