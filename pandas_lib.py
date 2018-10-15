import pandas as pd
import sqlite3
import numpy as np

from pprint import pprint as pp

pd.options.display.max_rows = 7  # Отображение количества строк
pd.set_option('display.max_columns', 100)  # Второй вариант настройки. Количесвто столбцов


# Работа с .loc
def work_loc(df: pd.DataFrame, column_for_row_check: str, any, column_for_show: str, func: function):
    return df.loc[(df[column_for_row_check] == any), column_for_show].apply(func)


# Посчитать количество повторений в столбце
def v_counts(column: pd.Series):
    return column.value_counts()


# Отобразить файл на графике
def chart(file: pd.DataFrame, kind: str):
    file.plot(kind=kind)


# Для работы с фреймом или сериями, чтобы не возникала ошибка копирования, вызывается copy
def no_copy_errors(data):
    return data.copy()


# Получить значение из даты, если дата - это индекс
def get_any_from_date(column: pd.Series):
    return column.index.day  # Где day можно заменить на вариации


# Сгруппировать данные по значениям определённой колонки, а для иных колонок провести расчёты
# самые популярные фукнции ['first', 'last', 'min', 'max', 'median', 'std', 'count', 'sum']
def group_and_calc(df: pd.DataFrame, column_grp: list, columns_nums: list, columns_str: list, funcs_nums: list, funcs_str: list):
    return df.groupby(column_grp)[columns_nums].agg(funcs_nums).join(df.groupby(column_grp)[columns_str].agg(funcs_str))
    # df[df.Year == 1996].groupby('Sex').agg('min')[Age]
    # df.groupby(['Medal', 'Sport']).get_group(('Silver', 'Tennis)).shape[0]


# Сгруппировать данные, если их индекс - дата. Например группировка по месяцу, или неделям
def group_by_indexdate(column: pd.Series, type: str):
    return column.resample(type)


# Редактирование элементов pd.Series
def change_by_symbol(columns: pd.Series, symbol: str, change: str):
    return columns.str.contains(symbol).fillna(change)


# Коннект к MySQL, SQLite, PostgreSQL или иным SQL базам
def sql_connect(directory: str, file_name: str, file_type: str, index_column: str):
    connect = sqlite3.connect(directory + '/' + file_name + file_type)
    return pd.read_sql('SELECT * from ' + file_name + ' LIMIT 3', connect, index_col=index_column)


# Просмотр общей информации по фрейму
def show_info(file: pd.DataFrame):
    return file.info()


# Содержание колонок или колонки
def all_column_info(df: pd.DataFrame):
    return df.describe()
    # df.describe().column_name


# Применение функции к каждому столбцу (или строке, если указать axis=1)
def make_for_all(df: pd.DataFrame, func_name: function):
    return df.apply(func_name)
    # column.apply(lambda x: x / sum(column)) Расчёт доли для каждого элемента в столбце


# Таблица сопряжённости, для поиска взаимосвязей
def conjugation_table(column1: pd.Series, column2:pd.Series):
    return pd.crosstab(column1, column2, normalize=True, margins=True)


# Сводная таблица
def piv_table(file: pd.DataFrame, index: list, columns_name: list, column_val: str, func: str):
    return file.pivot_table(index=index, columns=columns_name, values=column_val, aggfunc=func)


# Разбивка на подходящие числовые группы
def to_groups(column: pd.Series, groups: tuple):
    return pd.cut(column, groups, include_lowest=True, right=False)


# Таблица корреляции
def correlation(df: pd.DataFrame):
    return df.corr()


# Конвертирование бинарных строк в цифру. Можно взять совпадения, а можно изменённые данные
def bool_to_int(column: pd.Series):
    return pd.factorize(column)[0]


# В скольких строках встречается данный объект
def count_by_row(df: pd.DataFrame):
    return df.size()


# Скользящее окно
def rolling_pd(column, window: int):
    return column.rolling(window=window)


# Убрать данные, не проходящие по процентилю
def filter_by_percentile(df: pd.DataFrame, column: str, quant: float):
    return df[df[column] < df[column].quantile(quant)]


# Удалить дубликаты
def del_duplic(df: pd.DataFrame, columns: list):
    return df.drop_duplicates(subset=columns)


# Вернуть индекс максимаьного значения колонки
def max_index(df: pd.DataFrame, column: str):
    return df[column].idxmax()
