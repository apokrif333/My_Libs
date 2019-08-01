from tqdm import tqdm

import os


# Динамическое отображение текущего состояния итерации
def show_iteration(any_iter):
    tqdm(any_iter)


# Объединить данные так, чтобы получить один кортеж
def unique_set(data1, data2):
    return data1.union(data2)


# Красивое указание пути
def os_path(folder_1: str, folder_2: str, file_name: str):
    return os.path.join(folder_1, folder_2, file_name)


# Дебагер в Jupyter Notebook
# После возникновения ошибки, в следующей ячейке "%debug"
