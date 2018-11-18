from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.datasets import make_classification
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import json
import pandas as pd

# Удаляем низкую дисперсию
x_data_generated, y_data_generated = make_classification()
print(VarianceThreshold(.7).fit_transform(x_data_generated).shape)
print(VarianceThreshold(.85).fit_transform(x_data_generated).shape)
print(VarianceThreshold(.9).fit_transform(x_data_generated).shape)

# Сравниваем чистые данные, удалённую дисперисию и скоринг качества классификации(KBest)
x_data_varth = VarianceThreshold(.9).fit_transform(x_data_generated)
x_data_kbest = SelectKBest(f_classif, k=5).fit_transform(x_data_generated, y_data_generated)
kb_var = SelectKBest(f_classif, k=5).fit_transform(x_data_varth, y_data_generated)

print("LR: ", cross_val_score(LogisticRegression(), x_data_generated, y_data_generated,scoring='neg_log_loss').mean())
print("LR+var: ", cross_val_score(LogisticRegression(), x_data_varth, y_data_generated,scoring='neg_log_loss').mean())
print("LR+kbest", cross_val_score(LogisticRegression(), x_data_kbest, y_data_generated,scoring='neg_log_loss').mean())
print("LR+kbest+var", cross_val_score(LogisticRegression(), kb_var, y_data_generated,scoring='neg_log_loss').mean())

# Используем baseline-модель (Forest или линейная с лассо-регуляризацией) для оценки признаков, чтобы после неё
# передавать отобранные признаки в более сложную модель
x_data_generated, y_data_generated = make_classification()

lr = LogisticRegression()
rf = RandomForestClassifier()
pipe = make_pipeline(SelectFromModel(estimator=rf), lr)

print(cross_val_score(lr, x_data_generated, y_data_generated,scoring='neg_log_loss').mean())
print(cross_val_score(rf, x_data_generated, y_data_generated,scoring='neg_log_loss').mean())
print(cross_val_score(pipe, x_data_generated, y_data_generated,scoring='neg_log_loss').mean())

# Применим baseline на реальных данных. Не всегда байзлайн - это хорошая точка отправки
with open('C:/Users/Tom/PycharmProjects/Start/GibHub/My_Libs/test_data/train.json', 'r') as raw_data:
    data = json.load(raw_data)
    df = pd.DataFrame(data)

X_data = df[['bathrooms', 'bedrooms', 'price']]
y_data = df['interest_level']

pipe1 = make_pipeline(StandardScaler(), SelectFromModel(estimator=rf), lr)
pipe2 = make_pipeline(StandardScaler(), lr)

print('LR + selection: ', cross_val_score(pipe1, X_data, y_data, scoring='neg_log_loss').mean())
print('LR: ', cross_val_score(pipe2, X_data, y_data, scoring='neg_log_loss').mean())
print('Rf: ', cross_val_score(rf, X_data, y_data, scoring='neg_log_loss').mean())

# Exhaustive_Features_Selection
selector = SequentialFeatureSelector(lr, scoring='neg_log_loss', verbose=2, k_features=3, forward=False, n_jobs=4)
print(selector.fit(X_data, y_data))
