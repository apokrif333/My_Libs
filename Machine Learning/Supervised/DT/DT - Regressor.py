from sklearn.tree import DecisionTreeRegressor

import numpy as np
import matplotlib.pyplot as plt


# Генерируем рандом-семплы с некоторым шумом
def generate_samples_and_noise(n_samples, noise):
    X = np.random.rand(n_samples) * 10 - 5
    X = np.sort(X).ravel()
    y = np.exp(-X ** 2) + 1.5 * np.exp(-(X - 2) ** 2) + np.random.normal(0.0, noise, n_samples)
    X = X.reshape((n_samples, 1))
    return X, y


# Применяем к каждому из множества значений формулу
def func_for_elements(x: list):
    x = x.ravel()
    return np.exp(-x ** 2) + 1.5 * np.exp(-(x - 2) ** 2)


# DecisionTreeRegressor
# Применяем дерево решений на синтетических данных. Тут, для решения количественной классификации.
n_train = 150
n_test = 1_000
noise = 0.1

X_train, y_train = generate_samples_and_noise(n_samples=n_train, noise=noise)
X_test, y_test = generate_samples_and_noise(n_samples=n_test, noise=noise)

reg_tree = DecisionTreeRegressor(max_depth=5, random_state=17)
reg_tree.fit(X_train, y_train)
reg_tree_pred = reg_tree.predict(X_test)

plt.plot(X_test, func_for_elements(X_test), "b")
plt.scatter(X_train, y_train, c='b', s=20)
plt.plot(X_test, reg_tree_pred, 'g', lw=2)
plt.xlim([-5, 5])
plt.title('Decision tree regressor, MSE = %.2f' % np.sum((y_test - reg_tree_pred) ** 2))
plt.show()
