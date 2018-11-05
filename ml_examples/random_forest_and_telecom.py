import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score

plt.style.use('ggplot')


# Ищем оптимальный параметр через кросс-валидацию
def regularizer_finder(skf, X, y, rfc):
    temp_train_acc = []
    temp_test_acc = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        rfc.fit(X_train, y_train)
        temp_train_acc.append(rfc.score(X_train, y_train))
        temp_test_acc.append(rfc.score(X_test, y_test))

    train_acc.append(temp_train_acc)
    test_acc.append(temp_test_acc)


# Выводим кривые валидации
def plot_learning_curves(grid, train_acc, test_acc, x_name: str):
    train_acc, test_acc = np.asarray(train_acc), np.asarray(test_acc)
    print("Best accuracy on CV is {:.2F}% with {} {}".format(max(test_acc.mean(axis=1)) * 100,
                                                            grid[np.argmax(test_acc.mean(axis=1))],
                                                            x_name
                                                            ))

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(grid, train_acc.mean(axis=1), alpha=0.5, color='blue', label='train')
    ax.plot(grid, test_acc.mean(axis=1), alpha=0.5, color='red', label='cv')
    ax.fill_between(grid,
                    test_acc.mean(axis=1) - test_acc.std(axis=1),
                    test_acc.mean(axis=1) + test_acc.std(axis=1),
                    color='#888888',
                    alpha=0.4
                    )
    ax.fill_between(grid,
                    test_acc.mean(axis=1) - 2*test_acc.std(axis=1),
                    test_acc.mean(axis=1) + 2*test_acc.std(axis=1),
                    color='#888888',
                    alpha=0.2
                    )
    ax.legend(loc='best')
    ax.set_ylim([0.88, 1.02])
    ax.set_ylabel('Accuracy')
    ax.set_xlabel(x_name)


df = pd.read_csv('C:/Users/Tom/PycharmProjects/Start/GibHub/My_Libs/test_data/telecom_churn.csv')

cols = []
for i in df.columns:
    if (df[i].dtype == 'float64') or (df[i].dtype == 'int64'):
        cols.append(i)
X, y = df[cols].copy(), np.asarray(df['Churn'], dtype='int8')

# Инициализируем страифицированную разбивку нашего датасета для валидации
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rfc = RandomForestClassifier(random_state=42, n_jobs=-1, oob_score=True)
results = cross_val_score(rfc, X, y, cv=skf)
print(f"CV accuracy score: {results.mean()*100}")

# Построим кривые валидации для подбора количества деревьев
train_acc = []
test_acc = []
trees_grid = [5, 10, 15, 20, 30, 50, 75, 100]

for ntrees in trees_grid:
    rfc = RandomForestClassifier(n_estimators=ntrees, random_state=42, n_jobs=-1, oob_score=True)
    regularizer_finder(skf, X, y, rfc)
plot_learning_curves(trees_grid, train_acc, test_acc, 'n_estimators')

# Наша тренировочная модель показывает 100% результат, она переобучена. Найдём max_depth
train_acc = []
test_acc = []
max_depth_grid = [3, 5, 7, 9, 11, 13, 15, 17, 20, 22, 24]

for max_depth in max_depth_grid:
    rfc = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, oob_score=True,
                                 max_depth=max_depth)
    regularizer_finder(skf, X, y, rfc)
plot_learning_curves(max_depth_grid, train_acc, test_acc, 'max_depth')

# Теперь найдём минимальное количество элементов для дальнейшего разделения
train_acc = []
test_acc = []
min_samples_leaf_grid = [1, 3, 5, 7, 9, 11, 13, 15, 17, 20, 22, 24]

for min_samples_leaf in min_samples_leaf_grid:
    rfc = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, oob_score=True,
                                 min_samples_leaf=min_samples_leaf)
    regularizer_finder(skf, X, y, rfc)
plot_learning_curves(min_samples_leaf_grid, train_acc, test_acc, 'min_samples_leaf')

# И наконец найдём max_features. По умолчанию sqrt(n), где n, число признаков.
train_acc = []
test_acc = []
max_features_grid = [2, 4, 6, 8, 10, 12, 14, 16]

for max_features in max_features_grid:
    rfc = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, oob_score=True,
                                 max_features=max_features)
    regularizer_finder(skf, X, y, rfc)
plot_learning_curves(max_features_grid, train_acc, test_acc, 'max_features')

plt.show()

# Итак, мы нашли. n_estimators: 50, max_depth: 17, max_features: 12, min_samples_leaf: 3. В рамках найденных параметров
# с помощью GridSearchCV найдём оптимальные параметры для всей модели
parameters = {'max_features': [4, 7, 10, 12, 15], 'min_samples_leaf': [1, 3, 5, 7], 'max_depth': [5, 10, 15, 17, 20]}
rfc = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, oob_score=True)
gcv = GridSearchCV(rfc, parameters, n_jobs=-1, cv=skf, verbose=1).fit(X, y)
print(gcv)
