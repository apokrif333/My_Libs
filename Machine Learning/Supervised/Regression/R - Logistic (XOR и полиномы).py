from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import numpy as np
import matplotlib.pyplot as plt


# XOR проблема, т.е. исключающее ИЛИ, как проблема для линейных моделей
# Создаем дата-сет с исключающием или. Или одно, или другое.
rng = np.random.RandomState(0)  # Это подобие random_seed, только сохраняется на будущее, без вызова
X = rng.randn(200, 2)
print(X)
y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
plt.scatter(X[:, 0], X[:, 1], s=30, c=y, cmap=plt.cm.Paired)

# Попытаемся обучить линейной
clf = LogisticRegression()
plot_title = 'Logictic Regression, XOR problem'

xx, yy = np.meshgrid(np.linspace(-3, 3, 50), np.linspace(-3, 3, 50))
clf.fit(X, y)
# Прогнозируем вероятности
Z = clf.predict_proba(np.vstack((xx.ravel(), yy.ravel())).T)[:, 1]
Z = Z.reshape(xx.shape)

image = plt.imshow(Z, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
                   origin='lower', cmap=plt.cm.PuOr_r)
contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2, linetypes='--')
plt.scatter(X[:, 0], X[:, 1], s=30, c=y, cmap=plt.cm.Paired)
plt.xticks(())
plt.yticks(())
plt.xlabel(r'$<!--math>$inline$x_1$inline$</math -->$')
plt.ylabel(r'$<!--math>$inline$x_2$inline$</math -->$')
plt.axis([-3, 3, -3, 3])
plt.colorbar(image)
plt.title(plot_title, fontsize=12)
plt.show()

# Подадим полиноминальные признаки, т.е. создадим 6-мерное пространство
logit_pipe = Pipeline(
    [('poly', PolynomialFeatures(degree=2)),
     ('logit', LogisticRegression())]
)
clf = logit_pipe
plot_title = 'Logictic Regression + quadratic features. XOR problem'

xx, yy = np.meshgrid(np.linspace(-3, 3, 50), np.linspace(-3, 3, 50))
clf.fit(X, y)
# Прогнозируем вероятности
Z = clf.predict_proba(np.vstack((xx.ravel(), yy.ravel())).T)[:, 1]
Z = Z.reshape(xx.shape)

image = plt.imshow(Z, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
                   origin='lower', cmap=plt.cm.PuOr_r)
contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2, linetypes='--')
plt.scatter(X[:, 0], X[:, 1], s=30, c=y, cmap=plt.cm.Paired)
plt.xticks(())
plt.yticks(())
plt.xlabel(r'$<!--math>$inline$x_1$inline$</math -->$')
plt.ylabel(r'$<!--math>$inline$x_2$inline$</math -->$')
plt.axis([-3, 3, -3, 3])
plt.colorbar(image)
plt.title(plot_title, fontsize=12)
plt.show()
