from tqdm import tqdm


# Динамическое отображение текущего состояния итерации
def show_iteration(any_iter):
    tqdm(any_iter)


# Объединить данные так, чтобы получить один кортеж
def unique_set(data1, data2):
    return data1.union(data2)