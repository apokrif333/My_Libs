from __future__ import division, print_function
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score, TimeSeriesSplit, StratifiedShuffleSplit, RandomizedSearchCV
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, Imputer, OneHotEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from sklearn.externals import joblib
from scipy import stats
from graphviz import render
from typing import Callable

import pandas as pd
import numpy as np


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
def cross_valid_tree(des_tree_params, depth: list, features: list, cv_samples: int, random: int, X_train, y_train):
    skf = StratifiedKFold(n_splits=cv_samples, shuffle=False, random_state=random)
    return GridSearchCV(des_tree_params, max_depth=depth, max_features=features, cv=skf, n_jobs=-1,
                        verbose=True).fit(X_train, y_train)


# Классифицируем данные методом ближайших соседей
def kNN_create(nighbors: int, X_train, y_train):
    return KNeighborsClassifier(n_neighbors=nighbors).fit(X_train, y_train)


# Обучаем соседей по кросс-валидации
def cross_valid_kNN(neighbors: list, cv_samples: int, X_train, y_train):
    knn_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_jobs=-1))
    ])
    return GridSearchCV(knn_pipe, knn__n_neighbors=neighbors, cv=cv_samples, n_jobs=-1, verbose=True).fit(X_train,
                                                                                                          y_train)


# Веса признаков у случайного лесаz
def rforest_features(train_forest):
    return train_forest.feature_importances_


# Создаём матрицу из текстового файла, для анализа токенов-слов
def words_tokens(range: tuple, max: int, train_text):
    cv = CountVectorizer(ngram_range=range, max_features=max)
    vectorizer = make_pipeline(cv, TfidfTransformer())
    return vectorizer.fit_transform(train_text)


# Выводим все значения для настройки естиматора
def get_all_params(estimator):
    print(estimator.get_params().keys())


# Создаём dot-file c обученным деревом
def create_dot(clf_tree, feature_names: list, file_name: str, clases: list):
    export_graphviz(clf_tree, feature_names=feature_names, out_file=file_name + '.dot', filled=True,  class_names=clases)


# Конвертируем dot в png
def dot_to_png(name: str):
    path = 'img/' + name + '.dot'
    render('dot', 'png', path)


# Features ------------------------------------------------------------------------------------------------------------
# Заменить пустые значения на...
def change_empty(df: pd.DataFrame, strata: str):
    imputer = Imputer(strategy=strata)
    imputer.fit(df)
    print(imputer.statistics_)
    X = imputer.transform(df)
    return pd.DataFrame(X, columns=df.columns)


# Скалируем признак
def feature_scaler(column: list):
    return StandardScaler().fit_transform(column)


# Конвертация значений столбца в бинарные признаки
def encoder():
    preprocessing.LabelEncoder()


# OneHotEncoding. Категориальные значения в двоичные признаки (dummy)
def OneHotEncoding(series: pd.Series):
    encoder = OneHotEncoder()
    series = encoder.fit_transform(series)
    print(encoder.categories_)
    return series.toarray()


# Пайп-лайн который работает с определёнными колонками и может обращаться к обычным пайп-лайнам
def columns_pipeline(df: pd.DataFrame, numeric_clms: list, categoric_clms: list, num_pipe, cat_pipe):
    full_pipeline = ColumnTransformer([
        ("num", num_pipe, numeric_clms),
        ("cat", cat_pipe, categoric_clms),
    ])
    print(full_pipeline.named_transformers_['cat'].categories_[0])
    return full_pipeline.fit_transform(df)


# GridSeach для оптимизации гиперпараметров предобработки данных пайплайна
def grid_seach_for_pipeline(prepare_select_predict_pipeline, features_names: list, X, y):
    param_grid = [{
        'preparation__num__imputer_strategy': ['mean', 'median', 'most_frequent'],
        'features_selection__k': list(range(1, len(features_names) + 1))
    }]
    g_search = GridSearchCV(prepare_select_predict_pipeline, param_grid, cv=3,
                            scoring='neg_mean_squared_error', verbose=2)
    g_search.fit(X, y)
    return g_search.best_params_


# Получение нового нампи-фрейма с самыми эффективными признаками
def df_with_best_features(X: np.array, feat_importance: np.array, count_best_feat: int, feat_names: list):
    # Сортируем значения от меньшего к большему, распределяем значения так, чтобы все значения, которые меньше значения
    # под индексом count_best_features, были перед ним, а большие за ниим. Возвращаем индексы значений.
    best_features_indexes = np.sort(np.argpartition(feat_importance, -count_best_feat)[-count_best_feat:])
    print('Best features names: ', np.array(feat_names)[best_features_indexes])
    print(sorted(zip(feat_importance, feat_names), reverse=True)[:count_best_feat])

    return X[:, best_features_indexes], np.array(feat_names)[best_features_indexes]


# Regression ----------------------------------------------------------------------------------------------------------
# Используем последовательный тайм-срез через грид-сёрч, чтобы найти лучший регрессор в логист. регрессии
def time_split_cv_for_logit(splits: int, random: int, c_start: int, c_end: int, c_step: int, X, y):
    time_split = TimeSeriesSplit(n_splits=splits)
    logit = LogisticRegression(C=1, random_state=random)
    c_values = np.logspace(c_start, c_end, c_step)
    logit_grid_searcher = GridSearchCV(estimator=logit, param_grid={'C': c_values},
                                       scoring='roc_auc', n_jobs=4, cv=time_split, verbose=1)
    logit_grid_searcher.fit(X, y)
    return logit_grid_searcher.best_score_, logit_grid_searcher.best_params_


# Получение коэффициентов формулы регрессии
def regres_formula_values(reg_fit):
    return reg_fit.intercept_, reg_fit.coef_


# Оценка полученой логистичекой регрессии
def estimate_logit(logit_grid, X_test, y_test):
    test = logit_grid.predict_proba(X_test)[:, 1]
    print(roc_auc_score(y_test, test))
    return test


# Доверительный интервал для ошибок регрессии
def confidence_interval_for_errors(conf: float, y_pred, y):
    squared_errors = (y_pred - y) ** 2
    print(stats.t.interval(conf, len(squared_errors) - 1, loc=squared_errors.mean(), scale=stats.sem(squared_errors)))


# Trees ---------------------------------------------------------------------------------------------------------------

# Useful for all models -----------------------------------------------------------------------------------------------
# Создание отложенной выборки X_train, X_holdout, y_train, y_holdout
def hold_out_create(df: pd.DataFrame, y: pd.Series, test: float, random: int):
     return train_test_split(df.values, y, test_size=test, random_state=random)


# Стратифицируем базу данных. Разобъём train и test так, чтобы сохранить распредление данных, как у ведущего признака
def stratified_dataframe(df: pd.DataFrame, main_feature: str, test_size: float):
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    for train_index, test_index in split.split(df, df[main_feature]):
        strat_train_set = df.loc[train_index]
        strat_test_set = df.loc[test_index]
    return strat_train_set, strat_test_set


# Пайплайн для полиномизации, скалирования, модели и теста
def pipe_for_poly_scal_model(poly_degree: int, any_model: Callable, X, y, X_test):
    poly = preprocessing.PolynomialFeatures(degree=poly_degree)
    scaler = preprocessing.StandardScaler()
    model = any_model()
    pipeline = Pipeline([
        ('poly', poly),
        ('scal', scaler),
        ('model', model)
    ])
    pipeline.fit(X, y)
    return pipeline.predict(X_test)


# Оценка кросс-валидации по средней c созданием кросс-валидации
def cv_mean(trained_model, X_train, y_train, cv_samples: int):
    scores = cross_val_score(trained_model, X_train, y_train, scoring="neg_mean_squared_error", cv=cv_samples)
    rmse = np.sqrt(-scores)
    print('Scores: ', rmse)
    print('Mean: ', rmse.mean())
    print('St deviation: ', rmse.std())


# Выводим качество кросс-валидации
def cv_quality(model_grid, y_holdout, X_holdout):
    print(pd.DataFrame(model_grid.cv_results_))
    print(model_grid.best_params_, model_grid.best_score_)
    print(accuracy_score(y_holdout, model_grid.predict(X_holdout)))


def grid_search(model, cross_valid: int, X, y) -> pd.DataFrame:
    # Проведём сразу два поиска
    param_grid = [
        {'n_estimators': [3, 10, 32], 'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
    ]
    g_search = GridSearchCV(model, param_grid, cv=cross_valid,
                            scoring='neg_mean_squared_error', return_train_score=True)
    g_search.fit(X, y)

    print(g_search.best_params_)
    print(g_search.best_estimator_)
    print(g_search.best_estimator_.feature_importances_)
    for mean_score, params in zip(g_search.cv_results_['mean_test_score'], g_search.cv_results_['params']):
        print(np.sqrt(-mean_score), params)

    return pd.DataFrame(g_search.cv_results_)


def random_search(model, cross_valid: int, iterations: int, random_state: int, X, y):
    param_distribs = {
        'n_estimators': stats.randint(low=1, high=200),
        'max_features': stats.randint(low=1, high=8),
        'C': stats.reciprocal(20, 200000),
        'gamma': stats.expon(scale=1.0)
    }
    r_search = RandomizedSearchCV(model, param_distributions=param_distribs, n_iter=iterations, cv=cross_valid,
                                  scoring='neg_mean_squared_error', random_state=random_state)
    r_search.fit(X, y)

    for mean_score, params in zip(r_search.cv_results_['mean_test_score'], r_search.cv_results_['params']):
        print(np.sqrt(-mean_score), params)

    return r_search.best_estimator_


# Предсказываем данные по обученной модели
def model_predict(model, X_holdout):
    return model.predict(X_holdout)


# Выводим аккураси
def print_accuracy(y_holdout, X_holdout):
    print(accuracy_score(y_holdout, X_holdout))


# Ошибка RMSE
def rmse_error(real_labels: list, model_predictions: list):
    return np.sqrt(mean_squared_error(real_labels, model_predictions))


# Дамп обученной модели в файл
def trained_model_to_file(model, path: str, name: str):
    joblib.dump(model, path + name + '.pkl')


# Загрузка обученной модели
def trained_model_from_file(model, path: str, name: str):
    return joblib.load(model, path + name + '.pkl')
