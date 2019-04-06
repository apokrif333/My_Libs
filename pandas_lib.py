from typing import Callable

import pandas as pd
import pickle
import sqlite3
import numpy as np

pd.options.display.max_rows = 7  # Отображение количества строк
pd.set_option('display.max_columns', 100)  # Второй вариант настройки. Количесвто столбцов


# Start for dataframe research ----------------------------------------------------------------------------------------
# Просмотр общей информации по фрейму
def show_info(file: pd.DataFrame):
    return file.info()


# Содержание колонок или колонки
def all_column_info(df: pd.DataFrame):
    return df.describe()
    # df.describe().column_name


# Посчитать количество повторений в столбце
def v_counts(column: pd.Series):
    return column.value_counts()


# Тип данных колонок
def columns_data(df: pd.DataFrame):
    return df.dtypes


# Working with data ---------------------------------------------------------------------------------------------------
# Работа с .loc
def work_loc(df: pd.DataFrame, column_for_row_check: str, any, column_for_show: str, func: Callable):
    return df.loc[(df[column_for_row_check] == any), column_for_show].apply(func)


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


# Применение функции к каждому столбцу (или строке, если указать axis=1)
def make_for_all(df: pd.DataFrame, func_name: Callable[[int], int]):
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
def correlation(df: pd.DataFrame, matrix_column: str):
    return df.corr()[matrix_column].sort_values(ascending=False)


# Конвертирование класса значенний в цифру. Каждому уникальному значению присваивается своя цифра.
def class_to_int(column: pd.Series):
    return pd.factorize(column)


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


# Фиктивное кодирование. Создание доп. столбцов, которые расшифровывают наличие/неналичие элемента для признака
def dummy_encoding(df: pd.DataFrame, columns: list, prfx: str) -> pd.DataFrame:
    return pd.get_dummies(df, columns=columns, prefix=prfx)


# Создание нового столбца фильтруя по условиям текущий столбец
def filter_by_column(column: pd.Series, x_1: int, x_2: int) -> pd.Series:
    return column.apply(lambda x: 1 if x >= x_1 and x < x_2 else 0)


# Переименовать колонку
def rename_column(df: pd.DataFrame, old_name: str, new_name: str):
    df.rename(columns={old_name: new_name})


# Заменить заначения, которые удовлетворяют условию
def change_values(series: pd.Series):
    return series.where(series > 0, "yes")


# pickle
# ---------------------------------------------------------------------------------------------------------------------
def dumps_decode(frame):
    frame_dump = pickle.dumps(frame, protocol=0)
    return frame_dump.decode()


# Перевод серии в поток байтов и сохранение в файл
def frame_to_bite(name: str, frame):
    with open(name + '.pickle', 'wb') as f:
        pickle.dump(frame, f)


# Загрузка файла байт-потока
def bite_to_frame(name: str, frame):
    with open(name + '.pickle', 'rb') as f:
        return pickle.load(f)
