from sklearn.datasets import load_boston
from sklearn.linear_model import LassoCV, RidgeCV, Lasso, Ridge
from sklearn.model_selection import cross_val_score, KFold
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def lasso_reg():
    lasso = Lasso(alpha=0.1)
    lasso.fit(X, y)
    print(lasso.coef_)  # NOX-фича равна нулю. Она не оказывает эффекта на модель.

    # Так мы можем увидеть самые важные признаки. Доля занятой земли, коэффициент налога, коэффициент чёрных к белым,
    # процент маргиналов
    lasso = Lasso(alpha=10)
    lasso.fit(X, y)
    print(lasso.coef_)

    # Выведем влияние альфы (от наибольшей к наименьшей) для каждого признака
    n_alphas = 200
    alphas = np.linspace(0.1, 10, n_alphas)
    model = Lasso()

    coefs = []
    for a in alphas:
        model.set_params(alpha=a)
        model.fit(X, y)
        coefs.append(model.coef_)

    ax = plt.gca()
    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    plt.xlabel('alpha')
    plt.ylabel('weights')
    plt.title('Lasso coefficients as a function of the regularization')
    plt.axis('tight')
    plt.show()

    # Найдём оптимальную альфу
    lasso_cv = LassoCV(alphas=alphas, cv=3, random_state=17)
    lasso_cv.fit(X, y)
    print('------------')
    print(lasso_cv.coef_, "\n",
          lasso_cv.alpha_)

    # В Scikitlearn метрики обычно максимизируются, поэтому чтобы минимизировать их используется neg_mean_squeared_error
    print('------------')
    print(abs(cross_val_score(Lasso(lasso_cv.alpha_), X, y, cv=3, scoring='neg_mean_squared_error').mean()))
    print(abs(np.mean(cross_val_score(Lasso(9.95), X, y, cv=3, scoring='neg_mean_squared_error'))))

    # Полезно знать, что LassoCV сортирует значения параментров в порядке убывания
    print('------------')
    print(lasso_cv.alphas[:10])
    print(lasso_cv.alphas_[:10])
    plt.plot(lasso_cv.alphas, lasso_cv.mse_path_.mean(1))
    plt.axvline(lasso_cv.alpha_, c='g')
    plt.plot(lasso_cv.alphas_, lasso_cv.mse_path_.mean(1))
    plt.axvline(lasso_cv.alpha_, c='g')
    plt.show()


# Теперь исследуем гребневую регрессию
def ridge_reg():
    n_alphas = 200
    ridge_alphas = np.logspace(-2, 6, n_alphas)
    ridge_cv = RidgeCV(alphas=ridge_alphas, scoring='neg_mean_squared_error', cv=3)
    ridge_cv.fit(X, y)
    print('------------')
    print(ridge_cv.coef_, "\n",
          ridge_cv.alpha_)

    # Выведем влияние альфы (от наибольшей к наименьшей) для каждого признака
    model = Ridge()

    coefs = []
    for a in ridge_alphas:
        model.set_params(alpha=a)
        model.fit(X, y)
        coefs.append(model.coef_)

    ax = plt.gca()
    ax.plot(ridge_alphas, coefs)
    ax.set_xscale('log')
    ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    plt.xlabel('alpha')
    plt.ylabel('weights')
    plt.title('Ridge coefficients as a function of the regularization')
    plt.axis('tight')
    plt.show()


boston = load_boston()
X, y = boston['data'], boston['target']
print(boston.DESCR, "\n",
      boston.feature_names)
ridge_reg()