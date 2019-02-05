from plotly.offline import plot
from plotly import graph_objs as go

import warnings; warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plotly_df(df, title=''):
    data = []

    for column in df.columns:
        trace = go.Scatter(
            x=df.index,
            y=df[column],
            mode='lines',
            name=column
        )
        data.append(trace)

    layout = dict(title=title)
    fig = dict(data=data, layout=layout)
    plot(fig, show_link=False)


def moving_average(series, n):
    return np.average(series[-n:])


def plotMovingAverage(series, roll):
    rolling_mean = series.rolling(window=roll).mean()
    rolling_std = series.rolling(window=roll).std()

    upper_bond = rolling_mean + 1.96 * rolling_std
    lower_bond = rolling_mean - 1.96 * rolling_std

    plt.figure(figsize=(15, 5))
    plt.title('Moving average\n window size = {}'.format(roll))
    plt.plot(rolling_mean, 'g', label='Rolling mean trend')
    plt.plot(upper_bond, 'r--', label='Upper Bond / Lower Bond')
    plt.plot(lower_bond, 'r--')
    plt.plot(dataset[roll:], label='Actual values')
    plt.legend(loc='upper left')
    plt.grid(True)


def weighted_average(series, weights):
    result = 0.0
    weights.reverse()
    for n in range(len(weights)):
        result += series[-n-1] * weights[n]
    return result


def exponential_smoothing(series, alpha):
    result = [series[0]]
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return result


def double_exponential_smoothing(series, alpha, beta):
    result = [series[0]]

    for n in range(1, len(series)+1):

        if n ==1:
            level, trend = series[0], series[1] - series[0]
        if n >= len(series):
            value = result[-1]
        else:
            value = series[n]

        last_level, level = level, alpha * value + (1 - alpha) * (level + trend)
        trend = beta * (level - last_level) + (1 - beta) * trend
        result.append(level + trend)

    return result


dataset = pd.read_csv('data/hour_online.csv', index_col=['Time'], parse_dates=['Time'])
plotly_df(dataset, title='Online users')
print(moving_average(dataset.Users, 24))

plotMovingAverage(dataset, 24)
plotMovingAverage(dataset, 24 * 7)
plt.show()

print(weighted_average(dataset.Users, [0.6, 0.2, 0.1, 0.07, 0.03]))

# Экспоненциальное сглаживание
with plt.style.context('seaborn-white'):
    for alpha in [0.3, 0.05]:
        plt.figure(figsize=(20, 8))
        plt.plot(exponential_smoothing(dataset.Users, alpha), label='Alpha {}'.format(alpha))
        plt.plot(dataset.Users.values, 'c', label='Actual')
        plt.legend(loc='best')
        plt.axis('tight')
        plt.title('Exponential Smoothing')
        plt.grid(True)
        plt.show()

# Двойное сглаживание
with plt.style.context('seaborn-white'):
    for alpha in [0.9, 0.02]:
        for beta in [0.9, 0.02]:
            plt.figure(figsize=(20, 8))
            plt.plot(
                double_exponential_smoothing(dataset.Users, alpha, beta),
                label='Alpha {}, beta {}'.format(alpha, beta)
            )
            plt.plot(dataset.Users.values, label='Actual')
            plt.legend(loc='best')
            plt.axis('tight')
            plt.title('Double Exponential Smoothing')
            plt.grid(True)
            plt.show()
