from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
# from demo import get_data это тестовый пример, так как этот фром не работает

import json
import pandas as pd

with open('data/train.json', 'r') as raw_data:
    data = json.load(raw_data)
    df = pd.DataFrame(data)

X_data = df[['bathrooms', 'bedrooms', 'price']]
y_data = df['interest_level']

X_data = X_data.values

print(cross_val_score(LogisticRegression(), X_data, y_data, scoring='neg_log_loss').mean())
print(cross_val_score(LogisticRegression(),
                      StandardScaler().fit_transform(X_data),
                      y_data,
                      scoring='neg_log_loss').mean())
print(cross_val_score(LogisticRegression(),
                      MinMaxScaler().fit_transform(X_data),
                      y_data,
                      scoring='neg_log_loss').mean())
