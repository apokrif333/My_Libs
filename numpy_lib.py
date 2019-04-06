import numpy as np
from scipy.sparse import csr_matrix
import time

from numpy import linalg
from pprint import pprint as pp

np.seed = 7  # Теперь будет генерация одних и тех же случайных чисел при запуске


# Куммулятивная сумма в виде массива
def cum_sum(list_):
    return np.cumsum(list_)


# Собрать разные массивы в один массив
def to_one_massive(list_: list):
    return list.flatten()


# Массив от и до, с равным шагом
def np_linspace(start: int, end: int, steps: int) -> np.array:
    return np.linspace(start, end, steps)


# Массив со случаными данными, но распределённые по Гауссу
def gaussian_massive(rows: int, columns: int, centre: int):
    return np.random.normal(size=(rows, columns), loc=centre)


# Собрать разные листы или их срезы в один лист, но не складывать как обычно делает numpy
def one_list_for_many_lists(*list_: list):
    return np.r_[list_]


# Собрать лист с листами в один лист
def one_list_for_list(list_: list):
    return list.ravel()


# Транспонировать входящие массивы
def trans_lists(*list_: list):
    return np.c_[list_]


# Объединить входящие массивы в один
def unite_to_list(*list_: list):
    return np.r_[list_]

# Получение двух массивов x и y, и создание матрицы их положения, в стиле x[0, 20]=0, y[0, 20]=20
def x_y_matrix(x: list, y:list):
    return np.meshgrid(x, y)


# Вернуть знак числа (который, например, неизвестен до расчётов)
def np_sign(x):
    return np.sign(x)


# Вычислить значение истины по условию
def try_or_false(list_) -> list:
    return np.logical_xor(list_[:, 0] > 0, list_[:, 1] > 0)

# Случаный массив и заданных чисел
def random_from_list(data: list, size, probability: list):
    return np.random.choice(data, size=size, p=probability)


# Стак массивов по столбцам
def stack_by_columns(a: np.array, b: np.array):
    return np.hstack(a, b)


# Нампи, работа с датами
def np_date(int_: int, period: str):
    return np.timedelta64(int_, period)


# Разряжённая матрица. Матрица, которая хранит в себе значения, но преимущественно состоит из нулей, которые косвенно
# записаны в памяти, для экономии места.
# values_massive - одномерный массив значений, каждое уникальное занчение которого станет признаком.
# step - шаг, после которого сменяется строчка наблюдений.
# csr_matrix().todense() - вернуть разряжённой матрице обычный вид
def sparse_matrix(values_massive, step: int):
    return csr_matrix(([1] * values_massive.shape[0], values_massive, range(0, values_massive.shape[0] + step, step)))[
           :, 1:]


# Персентили
def np_persentile(list_, perc1: int, prec2: int):
    return np.percentile(list_, [perc1, prec2])


# Конвертация входящих данных в массив с определённым типом
def array_to_type(data, type: str):
    return np.asarray(data, dtype=type)


def random_txt_file():
    t1 = time.time()
    with open('Temp.txt', 'w') as f:
        for _ in range(4_000_000):
            f.write(str(10 * np.random.random()) + ',')
    t2 = time.time()
    print(f'Time for create file by  is {t2 - t1} seconds')


def random_npy_file():
    np_array = np.random.randint(0, 10, (4, 5, 6), dtype=int)
    np.save('Temp.npy', np_array)


def filter_by_column(file: np.ndarray, column: int, value: float) -> np.array:
    pp(file)
    return file[file[:, :, column] > value][:, np.array([False, False, True, False, True, False])]


def each_element(file: np.ndarray):
    for el in file.flat:
        pass


# -1 - количество измерений не определено. Оно определяет из оставшихся, заднных измерений
def cng_matrix_dimension(file: np.ndarray, columns: int) -> np.array:
    return file.reshape((-1, columns))


def massive_copy(file: np.ndarray) -> np.array:
    return file.copy()


def connect_str_to_str(file_1: np.array, file_2: np.array) -> np.array:
    return np.core.defchararray.add(file_1, file_2)


def sort_by_column(file: np.array, column: int):
    return file[file[:, column].argsort()]


if __name__ == '__main__':
    file = np.load('Temp.npy')

