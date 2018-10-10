import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly
import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

plt.style.use('ggplot')  # Красивые графики
plt.rcParams['figure.figsize'] = (20, 10)  # Размер картинок
plt.rcParams['font.family'] = 'sans-serif'  # Шрифт
init_notebook_mode(connected=True)


# ----------------------------------------------------------------------------------------------------------------------
# Matplotlib
# Вывести график
def chart_matplot(x, y):
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


# ----------------------------------------------------------------------------------------------------------------------
# Plotly
# График-линии
def chart_plotly_line(x: pd.Series, y1: pd.Series, y2: pd.Series, name1: str, name2: str, title: str):
    trace0 = go.Scatter(x=x, y=y1, name=name1)
    trace1 = go.Scatter(x=x, y=y2, name=name2)
    fig = go.Figure(data=[trace0, trace1], layout={'title': title})
    iplot(fig, show_link=False)


# График-бар
def chart_plotly_bar(x: pd.Series, y1: pd.Series, y2: pd.Series, name1: str, name2: str, title: str, s_title: str):
    trace0 = go.Bar(x=x, y=y1, name=name1)
    trace1 = go.Bar(x=x, y=y2, name=name2)
    fig = go.Figure(data=[trace0, trace1], layout={'title': title, 'xaxis': {'title': s_title}})
    iplot(fig, show_link=False)


# График с усами
def chart_plotly_box(df: pd.DataFrame, column_x: str, values_column: str):
    data = []
    for x in df[column_x].unique():
        data.append(go.Box(name=x, y=df[df[column_x] == x][values_column]))
    iplot(data, show_link=False)


# Сохранить в html график
def chart_html_save(figure, filename: str):
    plotly.offline.plot(figure, filename=filename + '.html', show_link=False)