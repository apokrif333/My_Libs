import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.style.use('ggplot')  # Красивые графики
plt.rcParams['figure.figsize'] = (20, 10)  # Размер картинок
plt.rcParams['font.family'] = 'sans-serif'  # Шрифт


# Вывести график
def charts(x, y):
    plt.plot(x, y, )


# Вывести два графика на основе данных, в которых два столбца и единый индекс
def two_charts(data, kind: str):
    data.plot(kind=kind, rot=45, subplots=True, figsize=(15, 10))


# ----------------------------------------------------------------------------------------------------------------------
# Seaborn
# Матрица колонок к друг другу. Диагональ разбивает значение одной колонки на децели, и показывает насыщение каждого
# дециля значениями из данной колонки
def charts_matrix_save(file: pd.DataFrame, path: str, name: str):
    sns.pairplot(file).savefig(path + '/' + name + '.png')


# Децильное рапределение набора данных
def decile_data(column: pd.Series):
    sns.distplot(column)


# Децили и взимосвязь двух данных
def decile_interchange(column1: pd.Series, column2: pd.Series):
    sns.jointplot(column1, column2)


# Ящик с усами (выбросы, условные 0,35%-25%, конец первого квартиля, медиана, конец третьего квартиля, условные
# 75%-99,65%, выборсы)
def box_plot(x: str, y: str, data: pd.DataFrame, orient):
    sns.boxplot(x=x, y=y, data=data, orient=orient)


# Тепловая карта для сводной таблицы
def heat_map(pivot_table):
    sns.heatmap(pivot_table, annot=True, fmt='.1f', linewidths=0.5)
