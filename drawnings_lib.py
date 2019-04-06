import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly
import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from pandas.plotting import scatter_matrix

plt.style.use('ggplot')  # Красивые графики
plt.rcParams['figure.figsize'] = (20, 10)  # Размер картинок
plt.rcParams['font.family'] = 'sans-serif'  # Шрифт
init_notebook_mode(connected=True)


# ----------------------------------------------------------------------------------------------------------------------
# Matplotlib
# Вывести график
def chart_matplot(x: pd.Series, y: pd.Series, title: str):
    plt.plot(x, y, "r--")
    plt.legend()
    plt.title(title)


# Точечный график
def scatter_plot(df: pd.DataFrame, x: str, y: str, size_column: str, color_column: str):
    df.plot(kind="scatter", x=x, y=y, alpha=0.4,
            s=df[size_column] / 100, label=size_column, figsize=(10, 7),
            c=color_column, cmap=plt.get_cmap("jet"), colorbar=True,
            sharex=False)
    plt.legend()


# Вывести два графика на основе данных, в которых два столбца и единый индекс
def two_charts(df: pd.DataFrame, kind: str):
    df.plot(kind=kind, rot=45, subplots=True, figsize=(15, 10))


# Вывести децильные гистограммы по каждому массиву данных
def decile_for_each(df: pd.DataFrame, columns_for_show: list, decile: int):
    df.hist(column=columns_for_show, bins=50, figsize=(20, 20))


# Тепловая карта
def heatmap_plt(pivot_table):
    plt.imshow(pivot_table, cmap='seismic', interpolation='none')


# Выделить контур
def contour_plt(x: list, y: list, z: list):
    plt.contour(x, y, z)


# График со стрелочками и текстом. В данном примере, названия стран служат индексом фрейма
def base_plot_with_arrows(df: pd.DataFrame, x: str, y: str, pos_text: dict):
    df.plot(kind='scatter', x=x, y=y, figsize=(10, 10))
    plt.axis([0, 60_000, 0, 10])
    position_text = {
        'Hungary': (5_000, 1),
        'Korea': (18_000, 1.7)
    }
    for country, pos_text in position_text.items():
        pos_data_x, pos_data_y = df.loc[country]
        plt.annotate(country, xy=(pos_data_x, pos_data_y), xytext=pos_text,
                     arrowprops=dict(facecolor='black', width=.5, shrink=.1, headwidth=5))
        plt.plot(pos_data_x, pos_data_y, 'ro')


# Матрица корреляции
def corr_matrix(df: pd.DataFrame, features: list):
    scatter_matrix(df[features], figsize=(20, 20))


# ----------------------------------------------------------------------------------------------------------------------
# Seaborn
# Матрица колонок к друг другу. Диагональ разбивает значение одной колонки на децели, и показывает насыщение каждого
# дециля значениями из данной колонки
def charts_matrix_save(file: pd.DataFrame, path: str, name: str):
    sns.pairplot(file).savefig(path + '/' + name + '.png')


# Диаграмма
def diargam_sns(x: str, y: str, df:pd.DataFrame):
    sns.countplot(x=x, hue=y, data=df)


# Децильное рапределение набора данных
def decile_data(column: pd.Series):
    sns.distplot(column)


# Децили и взимосвязь двух данных
def decile_interchange(column1: pd.Series, column2: pd.Series):
    sns.jointplot(column1, column2)


# Ящик с усами (выбросы, условные 0,35%-25%, конец первого квартиля, медиана, конец третьего квартиля, условные
# 75%-99,65%, выборсы)
def box_plot(x: str, y: str, df: pd.DataFrame, orient):
    sns.boxplot(x=x, y=y, data=df, orient=orient)


# Скрипка-график и ящик с усами в одном пространстве
def violin_box_plot(x: str, y: str, df: pd.DataFrame, orient):
    _, axes = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(16, 6))
    sns.boxplot(x=x, y=y, data=df, ax=axes[0])
    sns.violinplot(x=x, y=y, data=df, ax=axes[1])


# Создаём множество субграфиков, как пример, графики с усами
def sns_many_charts(nrows: int, ncols: int, column_anls: list, x: str, df: pd.DataFrame):
    fig, axes = plt.subplot(nrows=nrows, ncols=ncols, figsize=(18, 12))

    for idx, column in enumerate(column_anls):
        row = int(idx / ncols)
        column = int(idx % ncols)

        sns.boxplot(x=x, y=column, data=df, ax=axes[row, column])
        axes[row, column].legend()
        axes[row, column].set_xlabel(x)
        axes[row, column].set_ylabel(column)


# Тепловая карта для сводной таблицы
def heat_map(pivot_table):
    sns.heatmap(pivot_table, annot=True, fmt='.1f', linewidths=0.5)


# Распределение плотности значений
def sns_density(df: pd.DataFrame, column1: str, column_show: str, label:str, x_name: str, y_name: str):
    fig = sns.kdeplot(df[df[column1] == False][column_show], label=label)
    fig = sns.kdeplot(df[df[column1] == True][column_show], label=label)
    fig.set(xlabel=x_name, ylabel=y_name)


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
        data.append(go.Box(name=x, y=df[df[column_x] == x].values_column))
    iplot(data, show_link=False)


# Сохранить в html график
def chart_html_save(figure, filename: str):
    plotly.offline.plot(figure, filename=filename + '.html', show_link=False)
