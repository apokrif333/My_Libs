from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import beta, shapiro, lognorm
import statsmodels.api as sm

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Значение статистики. p-value
data = beta(1, 10).rvs(1_000).reshape(-1, 1)
print(shapiro(data))

print(shapiro(StandardScaler().fit_transform(data)))

# Скалирование защищает от выбросов. Скалирование по средней и отклонению
data = np.array([1, 1, 0, -1, 2, 3, -2, 4, 100]).reshape(-1, 1).astype(np.float64)
print('----------')
print(StandardScaler().fit_transform(data))
print((data - data.mean()) / data.std())

# Скалирование по MinMax
print('----------')
print(MinMaxScaler().fit_transform(data))
print((data-data.min()) / (data.max() - data.min()))

# Если данные распределены логнормально, то можно легко привести их к нормальному распределению
data = lognorm(s=1).rvs(1_000)
print('----------')
print(shapiro(data))
print(shapiro(np.log(data)))

# Построим графики логнорм. распределения и этого же распределения после логарифмирования
with open('data/train.json', 'r') as raw_data:
    data = json.load(raw_data)
    df = pd.DataFrame(data)

price = df.price[(df.price <= 2_000) & (df.price > 500)]
price_log = np.log(price)
price_mm = MinMaxScaler().fit_transform(price.values.reshape(-1, 1).astype(np.float64)).flatten()
price_z = StandardScaler().fit_transform(price.values.reshape(-1, 1).astype(np.float64)).flatten()

sm.qqplot(price_log, loc=price_mm.mean(), scale=price_log.std()).savefig('qq_price_log.png')
sm.qqplot(price_mm, loc=price_mm.mean(), scale=price_log.std()).savefig('qq_price_mm.png')
sm.qqplot(price_z, loc=price_mm.mean(), scale=price_log.std()).savefig('qq_price_z.png')
plt.show()
