

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Выдаём данные на график о средней ошибке, и её отклонении
def plot_with_err(x, data, **kwargs):
    mu, std = data.mean(1), data.std(1)
    lines = plt.plot(x, mu, '-', **kwargs)
    plt.fill_between(x, mu - std, mu + std, edgecolor='none', facecolor=lines[0].get_color(), alpha=0.2)


# Попытка понять, что лучше Простить/Усложнить модель, Добавить больше признаков, Накопить больше данных
data = pd.read_csv('data/telecom_churn.csv').drop('State', axis=1)
data['International plan'] = data['International plan'].map({'Yes': 1, 'No': 0})
data['Voice mail plan'] = data['Voice mail plan'].map({'Yes': 1, 'No': 0})
y = data['Churn'].astype('int').values
X = data.drop('Churn', axis=1).values

# Используем стохастический град. спуск и построим валид. кривые по обучающей и проверочной выборке
alphas = np.logspace(-2, 0, 20)  # коэффициеты регуляризации
sgd_logit = SGDClassifier(loss='log', n_jobs=-1, random_state=17)
logit_pipe = Pipeline(
    [('scaler', StandardScaler()), ('poly', PolynomialFeatures(degree=2)), ('sgd_logit', sgd_logit)])
val_train, val_test = validation_curve(logit_pipe, X, y, 'sgd_logit__alpha', alphas, cv=5, scoring='roc_auc')

# Обученные данные выводим на график. Тренировочная.
x = alphas
data = val_train
plot_with_err(alphas, val_train, label='training scores')

# Обученные данные выводим на график. Тестировочная.
data = val_test
plot_with_err(alphas, val_test, label='validation scores')

plt.xlabel(r'$\alpha$'); plt.ylabel('ROC AUC')
plt.legend()
plt.show()

# Результат нас не удовлетворяет. Построим кривые обучения, чтобы понять, нужно ли нам накопить больше данных
degree = 2
alpha = 10  # коэфф. регуляризации. Для примера взято значение 10.
train_sizes = np.linspace(0.05, 1, 20)
logit_pipe = Pipeline([('scalae', StandardScaler()), ('poly', PolynomialFeatures(degree=degree)),
                       ('sgd_logit', SGDClassifier(n_jobs=-1, random_state=17, alpha=alpha))])
N_train, val_train, val_test = learning_curve(logit_pipe, X, y, train_sizes=train_sizes, cv=5, scoring='roc_auc')
plot_with_err(N_train, val_train, label='trainings scores')
plot_with_err(N_train, val_test, label='validation scores')
plt.xlabel('Training Set Size'); plt.ylabel('AUC')
plt.legend()
plt.show()

# Как видно из кривых обучения, для небольшого объёма данных ошибки сильно отличаются, так как на небольшом объёме
# быстро возникает переобучение.
# А для больших объёмов, ошибки сходятся, что указывает на недообучение.
# Вывод: если добавить ещё данных, ошибка на обучающей не будет расти, но и на тестовых не будет уменьшаться. То
# есть ошибки сошлись и добавление новых данных не поможет.
# Значит надо улучшать модель. Исходя из первого графика видим, что мы можем взять коэфф. регуляризации 0.05
# Тогда мы видим, что кривые сходятся и ешё не стали параллельными, а значит в этом случае новые данные окажут
# эффект, чтобы повысить качество валидации.
# Если усложнить модель и выставить коэфф. регуляризации на 10 ** -4, то возникнет переобучение, качество модели
# постоянно падает, как на обучении, так и на валидации.
