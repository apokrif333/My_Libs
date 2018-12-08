from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression

import xgboost as xgb
import warnings; warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def code_mean(data, cat_feature, real_feature):
    """
    :param cat_feature: ключи для словаря
    :param real_feature: значения для словаря, средние
    :return: словарь, у которого ключи возникли из групп
    """
    return dict(data.groupby(cat_feature)[real_feature].mean())


def prepateData(data, lag_start=5, lag_end=20, test_size=0.15):
    data = pd.DataFrame(data.copy())
    data.columns = ['y']
    test_index = int(len(data) * (1 - test_size))

    # Лаги исходного ряда в качестве признаков
    for i in range(lag_start, lag_end):
        data[f"lag_{i}"] = data.y.shift(i)

    data.index = pd.to_datetime(data.index)
    data['hour'] = data.index.weekday
    data['weekday'] = data.index.weekday
    data['is_weekend'] = data.weekday.isin([5, 6]) * 1

    # Считаем среднюю только по тренировочной части, чтобы избежеть фиттинга
    data['weekday_average'] = list(map(code_mean(data[:test_index], 'weekday', 'y').get, data.weekday))
    data['hour_average'] = list(map(code_mean(data[:test_index], 'hour', 'y').get, data.hour))

    data.drop(['hour', 'weekday'], axis=1, inplace=True)
    data = data.dropna()
    data = data.reset_index(drop=True)

    # Разбиваем на тестовую и тренировочную выборки
    X_train = data.loc[:test_index].drop(['y'], axis=1)
    y_train = data.loc[:test_index]['y']
    X_test = data.loc[test_index:].drop(['y'], axis=1)
    y_test = data.loc[test_index:]['y']

    return X_train, X_test, y_train, y_test


def performTimeSeriesCV(X_train, y_train, number_folds, model, metrics):
    print(f"Size train set: {X_train.shape}")

    k = int(np.floor(float(X_train.shape[0]) / number_folds))
    print(f"Size of each fold: {k}")
    errors = np.zeros(number_folds - 1)

    for i in range(2, number_folds + 1):
        print('')
        split = float(i - 1) / i
        print('Splitting the first ' + str(i) + ' chunks at ' + str(i - 1) + '/' + str(i))

        X = X_train[:(k*i)]
        y = y_train[:(k * i)]
        print('Size of train + test: {}'.format(X.shape))

        index = int(np.floor(X.shape[0] * split))

        X_trainFolds = X[:index]
        y_trainFolds = y[:index]

        X_testFold = X[(index + 1):]
        y_testFold = y[(index + 1):]

        model.fit(X_trainFolds, y_trainFolds)
        errors[i-2] = metrics(model.predict(X_testFold), y_testFold)

    return errors.mean()


def XGB_forecast(data, lag_start=5, lag_end=20, test_size=0.15, scale=1.96):
    X_train, X_test, y_train, y_test = prepateData(dataset.Users, lag_start, lag_end, test_size)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)

    params = {'objective': 'reg:linear',
              'booster':'gblinear'}
    trees = 1_000

    # Кросс-валидация и метрика rmse
    cv = xgb.cv(params, dtrain, metrics=('rmse'), verbose_eval=False, nfold=10, show_stdv=False, num_boost_round=trees)
    bst = xgb.train(params, dtrain, num_boost_round=cv['test-rmse-mean'].argmin())

    # Кривые валидации и ошибка
    # cv.plot(y=['test-mae-mean', 'train-mae-mean'])
    deviation = cv.loc[cv['test-rmse-mean'].argmin(), 'test-rmse-mean']

    # Как модель вела себя на тренировочном отрезке ряда
    prediction_train = bst.predict(dtrain)
    plt.figure(figsize=(15, 5))
    plt.plot(prediction_train)
    plt.plot(y_train)
    plt.axis('tight')
    plt.grid(True)

    # Как модель вела себя на тестовом ряде
    prediction_test = bst.predict(dtest)
    lower = prediction_test - scale * deviation
    upper = prediction_test + scale * deviation

    Anomalies = np.array([np.NaN] * len(y_test))
    Anomalies[y_test<lower] = y_test[y_test<lower]

    plt.figure(figsize=(15, 5))
    plt.plot(prediction_test, label='prediction')
    plt.plot(lower, 'r--', label='upper bond / lower bond')
    plt.plot(upper, 'r--')
    plt.plot(list(y_test), label='y_test')
    plt.plot(Anomalies, 'ro', markersize=10)
    plt.legend(loc='best')
    plt.axis('tight')
    plt.title('XGBoost Mean absolute arror {} users'.format(round(mean_absolute_error(prediction_test, y_test))))
    plt.grid(True)
    plt.legend()


dataset = pd.read_csv('D:/Py_Projects/GitHub/My_Libs/ml_examples/test_data/hour_online.csv',
                      index_col=['Time'],
                      parse_dates=['Time'])

# Линейная регрессия
X_train, X_test, y_train, y_test = prepateData(dataset.Users, test_size=0.3, lag_start=12, lag_end=48)
lr = LinearRegression()
lr.fit(X_train, y_train)
prediction = lr.predict(X_test)

plt.figure(figsize=(15, 7))
plt.plot(prediction, 'r', label='prediction')
plt.plot(y_test.values, label='actual')
plt.legend(loc='best')
plt.title('Linerar regression\n Mean absolute error {} users'.format(round(mean_absolute_error(prediction, y_test))))
plt.grid(True)
plt.show()

print(performTimeSeriesCV(X_train, y_train, 5, lr, mean_absolute_error))

# XGBoost
XGB_forecast(dataset, test_size=0.2, lag_start=5, lag_end=30)
plt.show()
