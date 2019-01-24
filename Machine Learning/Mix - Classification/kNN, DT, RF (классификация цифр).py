from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import numpy as np
import matplotlib.pyplot as plt

# Обучим сеть разпознавать рукописные цифры деревом и соседями
# Загружаем дату, где цифры в виде матриц 8х8, каждое значение элемента - интенсивность белого
data = load_digits()
X, y = data.data, data.target
X[0, :].reshape([8, 8])

# Создадим в plt 4 суб-плота и в каждом выведем 4 первые цифры из даты
f, axes = plt.subplots(1, 4, sharey=True, figsize=(16, 6))
for i in range(4):
    axes[i].imshow(X[i, :].reshape([8, 8]))
plt.show()

# Создаём отложенную выборку, обучаем сети и прогнозируем выборку
X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.3, random_state=17)
tree = DecisionTreeClassifier(max_depth=5, random_state=17).fit(X_train, y_train)
knn = KNeighborsClassifier(n_neighbors=10).fit(X_train, y_train)
tree_pred = tree.predict(X_holdout)
knn_pred = knn.predict(X_holdout)
print(accuracy_score(y_holdout, tree_pred), accuracy_score(y_holdout, knn_pred))

# Создадим кросс-валидацию для дерева. Теперь признаков 64, так как цветовая модель из 8х8
tree_params = {
    'max_depth': [1, 2, 3, 5, 10, 20, 25, 30, 40, 50, 64],
    'max_features': [1, 2, 3, 5, 10, 20, 30, 50, 64]
}
tree_grid = GridSearchCV(tree, tree_params, cv=5, n_jobs=4, verbose=True).fit(X_train, y_train)
print(tree_grid.best_params_, tree_grid.best_score_)

# Один сосед и случайный лес
print(np.mean(cross_val_score(KNeighborsClassifier(n_neighbors=1), X_train, y_train, cv=5)))
print(np.mean(cross_val_score(RandomForestClassifier(random_state=17), X_train, y_train, cv=5)))
