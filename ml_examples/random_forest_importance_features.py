from __future__ import division, print_function

import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib import rc
from sklearn.ensemble.forest import RandomForestRegressor

warnings.filterwarnings('ignore')
font = {'family': 'Verdana',
        'weight': 'normal'}
rc('font', **font)

hostel_data = pd.read_csv('C:/Users/Lex/PycharmProjects/Start/GitHub/My_Libs/test_data/hostel_factors.csv')

features = {'f1': u"Персонал",
            'f2': u"Бронирование хостела",
            'f3': u"Заезд и выезд из хостела",
            'f4': u"Состояние комнаты",
            'f5': u"Состояние общей кухни",
            'f6': u"Состояние общего пространства",
            'f7': u"Дополнительные услуги",
            'f8': u"Общие условия и удобства",
            'f9': u"Цены/качество",
            'f10': u"ССЦ"
            }

forest = RandomForestRegressor(n_estimators=1_000, max_features=10, random_state=0)
forest.fit(hostel_data.drop(['hostel', 'rating'], axis=1), hostel_data['rating'])
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]
num_to_plot = 10
feature_indices = [ind+1 for ind in indices[:num_to_plot]]

print('Feature ranking: ')

for f in range(num_to_plot):
    print("%d. %s %f" %
          (f + 1,
           features["f" + str(feature_indices[f])],
           importances[indices[f]]
           ))

plt.figure(figsize=(15, 5))
plt.title(u"Важность конструкторов")
bars = plt.bar(range(num_to_plot), importances[indices[:num_to_plot]],
               color=([str(i/float(num_to_plot+1)) for i in range(num_to_plot)]),
               align='center')
ticks = plt.xticks(range(num_to_plot), feature_indices)
plt.xlim([-1, num_to_plot])
plt.legend(bars, [u''.join(features["f"+str(i)]) for i in feature_indices])
plt.show()