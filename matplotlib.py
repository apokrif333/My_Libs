import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')  # Красивые графики
plt.rcParams['figure.figsize'] = (20, 10)  # Размер картинок
plt.rcParams['font.family'] = 'sans-serif'  # Шрифт


# Вывести два графика на основе данных, в которых два столбца и единый индекс
def two_charts(data, kind: str):
    data.plot(kind=kind, rot=45, subplots=True, figsize=(15, 10))


# ----------------------------------------------------------------------------------------------------------------------
# Seaborn
