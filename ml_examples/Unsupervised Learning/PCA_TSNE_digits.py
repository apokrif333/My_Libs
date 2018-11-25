from sklearn import decomposition
from sklearn.manifold import TSNE
from sklearn import datasets

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style='white')

digits = datasets.load_digits()
X = digits.data
y = digits.target

# Картинки - матрицы 8х8 интенсивности каждого пикселя. Матрица разворачивается в вектор 64 для признакового описания.
# f, axes = plt.subplot(5, 2, sharey=True, figsize=(16, 6))
plt.figure(figsize=(16, 6))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X[i, :].reshape([8, 8]))
plt.show()

print('Projecting %d-dimensional data to 2D' % X.shape[1])
pca = decomposition.PCA(n_components=2)
X_reduced = pca.fit_transform(X)

plt.figure(figsize=(12, 10))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y,
            edgecolors='none', alpha=0.7, s=40,
            cmap=plt.cm.get_cmap('nipy_spectral', 10))
plt.colorbar()
plt.title('MNIST. PCA projection')
plt.show()

# Покажем, что имеет смысл сужать количество компонент так, чтобы оставалось не менее 90% исходной дисперсиии.
pca = decomposition.PCA().fit(X)

plt.figure(figsize=(10, 7))
plt.plot(np.cumsum(pca.explained_variance_ratio_), color='k', lw=2)
plt.xlabel('Number of components')
plt.ylabel('Total explained variance')
plt.xlim(0, 63)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.axvline(21, c='b')
plt.axhline(0.915, c='r')
plt.show()

# Применим метод стохастических соседей для сворачивания 64-признакового простанства до 2D.
tsne = TSNE(random_state=17)
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(12, 10))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y,
            edgecolors='none', alpha=0.7, s=40,
            cmap=plt.cm.get_cmap('nipy_spectral', 10))
plt.colorbar()
plt.title('MNIST. t-SNE projection')
plt.show()
