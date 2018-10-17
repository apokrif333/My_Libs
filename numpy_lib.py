import numpy as np
import time

from numpy import linalg
from pprint import pprint as pp

np.seed = 7  # Теперь будет генерация одних и тех же случайных чисел при запуске


# Массив от и до, с равным шагом
def np_linspace(start: int, end: int, steps: int) -> np.array:
    return np.linspace(start, end, steps)


# Массив со случаными данными, но распределённые по Гауссу
def gaussian_massive(rows: int, columns: int, centre: int):
    return np.random.normal(size=(rows, columns), loc=centre)


# Собрать разные листы или их срезы в один лист
def one_list_for_many_lists(*list_: list):
    return np.r_[list_]


# Собрать лист с листами в один лист
def one_list_for_list(list_: list):
    return list.ravel()


# Транспонировать входящие массивы
def trans_lists(*list_: list):
    return np.c_[list_]


# Получение двух массивов x и y, и создание матрицы их положения, в стиле x[0, 20]=0, y[0, 20]=20
def x_y_matrix(x: list, y:list):
    return np.meshgrid(x, y)


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


def auto_change_by_rows(file: np.ndarray, rows: int) -> np.array:
    return file.reshape((rows, -1))


def massive_copy(file: np.ndarray) -> np.array:
    return file.copy()


def connect_str_to_str(file_1: np.array, file_2: np.array) -> np.array:
    return np.core.defchararray.add(file_1, file_2)


def sort_by_column(file: np.array, column: int):
    return file[file[:, column].argsort()]


if __name__ == '__main__':
    file = np.load('Temp.npy')

