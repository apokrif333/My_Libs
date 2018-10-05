import pandas as pd
import sqlite3

from pprint import pprint as pp

pd.options.display.max_rows = 7  # Отображение количества строк


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


# Сгруппировать данные по значениям определённой колонки и вывести среднюю
def group_and_average(file: pd.DataFrame, column:str):
    return file.groupby(column).mean()


# Сгруппировать данные, если их индекс - дата. Например группировка по месяцу, или неделям
def group_by_indexdate(column: pd.Series, type: str):
    return column.resample(type)


# Редактирование элементов pd.Series
def change_by_symbol(columns: pd.Series, symbol: str, change:str):
    return columns.str.contains(symbol).fillna(change)


# Коннект к MySQL, SQLite, PostgreSQL или иным SQL базам
def sql_connect(directory: str, file_name: str, file_type: str, index_column: str):
    connect = sqlite3.connect(directory + '/' + file_name + file_type)
    return pd.read_sql('SELECT * from ' + file_name + ' LIMIT 3', connect, index_col=index_column)