from tqdm import tqdm
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.optimize import minimize
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly import graph_objs as go; init_notebook_mode(connected=True)

import sys
import warnings; warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import scipy.stats as scs


class HoltWinters:
    '''
    Модель Хольта-Винтерса с методом Брутлага для детектирования аномалий
    https://fedcsis.org/proceedings/2012/pliks/118.pdf

    series - исходный временной ряд
    slen - длина сезона
    alpha, beta, gamma - коэффициенты модели Хольта-Винтерса
    n_pred - горизонт предсказаний
    scaling_factor - задаёт ширину доверитального интервала по Брутлагу (обычно, от 2 до 3)
    '''

    def __init__(self, series, slen, alpha, beta, gamma, n_preds, scaling_factor=1.96):
        self.series = series
        self.slen = slen
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n_preds = n_preds
        self.scaling_factor = scaling_factor

    def initial_trend(self):
        sum = 0
        for i in range(self.slen):
            sum += float(self.series[i+self.slen] - self.series[i]) / self.slen
        return sum / self.slen

    def initial_seasonal_components(self):
        seasonals = {}
        season_averages = []
        n_seasons = int(len(self.series) / self.slen)

        # Вычислим сезонные средние
        for j in range(n_seasons):
            season_averages.append(sum(self.series[self.slen*j:self.slen*j+self.slen]) / float(self.slen))
        # Начальные значения
        for i in range(self.slen):
            sum_of_vals_over_avg = 0
            for j in range(n_seasons):
                sum_of_vals_over_avg += self.series[self.slen*j+i] - season_averages[j]
            seasonals[i] = sum_of_vals_over_avg / n_seasons

        return seasonals

    def triple_exponential_smoothing(self):
        self.result = []
        self.Smooth = []
        self.Season = []
        self.Trend = []
        self.PredictedDeviation = []
        self.UpperBond = []
        self.LowerBond = []

        seasonals = self.initial_seasonal_components()

        for i in range(len(self.series) + self.n_preds):
            if i == 0:
                smooth = self.series[0]
                trend = self.initial_trend()
                self.result.append(self.series[0])
                self.Smooth.append(smooth)
                self.Trend.append(trend)
                self.Season.append(seasonals[i%self.slen])

                self.PredictedDeviation.append(0)

                self.UpperBond.append(self.result[0] + self.scaling_factor * self.PredictedDeviation[0])
                self.LowerBond.append(self.result[0] - self.scaling_factor * self.PredictedDeviation[0])

                continue

            # Прогнозируем
            if i >= len(self.series):
                m = i - len(self.series) + 1
                self.result.append((smooth + m * trend) + seasonals[i%self.slen])
                # Увеличивает неопределённость с каждым шагомъ
                self.PredictedDeviation.append(self.PredictedDeviation[-1] * 1.01)

            else:
                val = self.series[i]
                last_smooth, smooth = smooth, \
                                      self.alpha * (val - seasonals[i%self.slen]) + (1 - self.alpha) * (smooth + trend)
                trend = self.beta * (smooth - last_smooth) + (1 - self.beta) * trend
                seasonals[i%self.slen] = self.gamma * (val - smooth) + (1 - self.gamma) * seasonals[i%self.slen]
                self.result.append(smooth + trend + seasonals[i%self.slen])

                # Отклонение рассчитывается в соотвествии с алгоритмом Брутлага
                self.PredictedDeviation.append(self.gamma * np.abs(self.series[i] - self.result[i]) +
                                               (1 - self.gamma) * self.PredictedDeviation[-1])

            self.UpperBond.append(self.result[-1] + self.scaling_factor * self.PredictedDeviation[-1])
            self.LowerBond.append(self.result[-1] - self.scaling_factor * self.PredictedDeviation[-1])
            self.Smooth.append(smooth)
            self.Trend.append(trend)
            self.Season.append(seasonals[i%self.slen])


def plotHolWinters(data):
    plt.figure(figsize=(25, 10))
    plt.plot(model.result, label='Model')
    plt.plot(data.values, label='Actual')
    # error = mean_squared_error(data.values, model.result[:len(data)])
    plt.title(f"Mean Squared Error: {error}")

    Anomalies = np.array([np.NaN]*len(data))
    Anomalies[data.values<model.LowerBond[:len(data)]] = data.values[data.values<model.LowerBond[:len(data)]]
    plt.plot(Anomalies, 'o', markersize=10, label='Anomalies')

    plt.plot(model.UpperBond, 'r--', alpha=0.5, label='Up/Low confidence')
    plt.plot(model.LowerBond, 'r--', alpha=0.5)
    plt.fill_between(x=range(0, len(model.result)), y1=model.UpperBond, y2=model.LowerBond, alpha=0.5, color='grey')

    plt.axvspan(len(data)-128, len(data), alpha=0.5, color='lightgrey')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='best', fontsize=13)
    plt.show()


def timeseriesCVscore(x):
    # Вектор ошибок
    errors = []

    values = data.values
    alpha, beta, gamma = x

    # Число фолдов для кросс-валидации
    tscv = TimeSeriesSplit(n_splits=3)

    # На каждом фолде обучаем нашу модель, прогнозим и считаем ошибку
    for train, test in tscv.split(values):
        model = HoltWinters(series=values[train], slen=24*7, alpha=alpha, beta=beta, gamma=gamma, n_preds=len(test))
        model.triple_exponential_smoothing()

        predictions = model.result[-len(test):]
        actual = values[test]
        error = mean_squared_error(predictions, actual)
        errors.append(error)

    # Средний квадрат ошибки по вектору ошибок
    return np.mean(np.array(errors))


#  C:/Users/Tom/PycharmProjects/Start/GibHub/My_Libs/ml_examples/test_data/hour_online.csv
dataset = pd.read_csv('D:/PycharmProjects/Start/GitHub/My_Libs/ml_examples/test_data/hour_online.csv',
                      index_col=['Time'],
                      parse_dates=['Time']
                      )
# model = HoltWinters(dataset.Users, 7, 0.05, 0.05, 0.05, 200)
# model.triple_exponential_smoothing()
# plt.figure(figsize=(20, 8))
# plt.plot(model.result, label=f"Season len {model.slen} Alpha {model.alpha} Beta {model.beta} Gamma {model.gamma}")
# plt.plot(dataset.Users.values, label='Actual')
# plt.legend(loc='best')
# plt.axis('tight')
# plt.title('Triple Exponential Smoothing')
# plt.grid(True)
# plt.show()

# Отложим часть данных для тестирования
data = dataset.Users[:-128]
# Инициализируем альфу, гаамму, бетту
x = [0, 0, 0]
# Минимизируем и ограничим функцию потерь
opt = minimize(timeseriesCVscore, x0=x, method='TNC', bounds=((0, 1), (0, 1), (0, 1)))
# Из оптимизиатора берём оптимальное значение
alpha_final, beta_final, gamma_final = opt.x
print(alpha_final, beta_final, gamma_final)

# Обучаем модель на найденых параметрах
model = HoltWinters(dataset.Users[:-128], slen=24*7, alpha = alpha_final, beta=beta_final, gamma=gamma_final,
                    n_preds=128, scaling_factor=2.56)
model.triple_exponential_smoothing()
plotHolWinters(dataset.Users)

