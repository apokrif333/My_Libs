from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

import os
import numpy as np
import matplotlib.pyplot as plt

# Определяем хороший отзыв о фильме или нет, при помощи логистической регресиии
path = 'data/aclImdb'
reviews_train = load_files(os.path.join(path, 'train'), categories=['pos', 'neg'])
text_train, y_train = reviews_train.data, reviews_train.target
reviews_test = load_files(os.path.join(path, 'test'), categories=['pos', 'neg'])
text_test, y_test = reviews_test.data, reviews_test.target
print(
    f"Number of documents in training data: {len(text_train)}\n",
    f"Number of documents in test data: {len(text_test)}\n",
    np.bincount(y_test)
)

# Создадим словарь всех слов с индексом
cv = CountVectorizer().fit(text_train)
print(len(cv.vocabulary_))
print(cv.get_feature_names()[:50])
print(cv.get_feature_names()[50_000:50_050])

X_train = cv.transform(text_train)
X_test = cv.transform(text_test)

# Обучим логист. регрессию
logit = LogisticRegression(n_jobs=-1, random_state=7).fit(X_train, y_train)
print(
    'LogitR without C on train and test: ' + str(round(logit.score(X_train, y_train), 3)),
    str(round(logit.score(X_test, y_test), 3))
)

# Тут должна быть фукнция, но вынесем её пока в тело. Отрисосываем веса ключевых слов после обучения
classifier = logit
feature_names = cv.get_feature_names()
n_top_features = 25
coef = classifier.coef_.ravel()
positive_coefficients = np.argsort(coef)[-n_top_features:]
negative_coefficients = np.argsort(coef)[:n_top_features]
interesting_coefficients = np.hstack([negative_coefficients, positive_coefficients])

plt.figure(figsize=(15, 5))
colors = ['red' if c < 0 else 'blue' for c in coef[interesting_coefficients]]
plt.bar(np.arange(2 * n_top_features), coef[interesting_coefficients], color=colors)
feature_names = np.array(feature_names)
plt.xticks(np.arange(1, 1 + 2 * n_top_features), feature_names[interesting_coefficients], rotation=60, ha='right')
plt.show()

# Подбираем коэфф. регулиризации для лог. регрессии. sklearn.pipeline говорит сначала применить CountVectorizer, а
# затем обучить регрессию. Так мы избегаем подсматривания в тестовую выборку.
text_pipe_logit = make_pipeline(CountVectorizer(), LogisticRegression(n_jobs=-1, random_state=7)).fit(text_train,
                                                                                                      y_train)
print('HO on test by train ' + str(text_pipe_logit.score(text_test, y_test)))

param_grid_logit = {'logisticregression__C': np.logspace(-5, 0, 6)}
grid_logit = GridSearchCV(text_pipe_logit, param_grid_logit, cv=3, n_jobs=-1).fit(text_train, y_train)
print('CV on train with C: ' + str(grid_logit.best_params_), str(grid_logit.best_score_))

grid = grid_logit
param_name = 'logisticregression__C'
plt.plot(grid.param_grid[param_name], grid.cv_results_['mean_train_score'], color='green', label='train')
plt.plot(grid.param_grid[param_name], grid.cv_results_['mean_test_score'], color='red', label='test')
plt.legend()
plt.show()

print('From CV on train to test: ' + str(grid_logit.score(text_test, y_test)))

# Сравним лог. регрессию и случ. лес
forest = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=17).fit(X_train, y_train)
print('RF from train to test: ' + str(round(forest.score(X_test, y_test), 3)))
