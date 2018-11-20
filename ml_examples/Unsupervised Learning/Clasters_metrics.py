from sklearn import metrics
from sklearn import datasets
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation, SpectralClustering
import pandas as pd; pd.options.display.max_columns = 20

data = datasets.load_digits()
X, y = data.data, data.target

algorithms = []
algorithms.append(KMeans(n_clusters=10, random_state=1))
algorithms.append((AffinityPropagation()))
algorithms.append(SpectralClustering(n_clusters=10, random_state=1, affinity='nearest_neighbors'))
algorithms.append(AgglomerativeClustering(n_clusters=10))

data =[]
for algo in algorithms:
    algo.fit(X)
    data.append(({'ARI': metrics.adjusted_rand_score(y, algo.labels_),
                  'AMI': metrics.adjusted_mutual_info_score(y, algo.labels_),
                  'Homogenity': metrics.homogeneity_score(y, algo.labels_),
                  'Completeness': metrics.completeness_score(y, algo.labels_),
                  'V-measure': metrics.v_measure_score(y, algo.labels_),
                  'Silhouette': metrics.silhouette_score(X, algo.labels_)
                  }))

results = pd.DataFrame(data=data, columns=['ARI', 'AMI', 'Homogenity', 'Completeness', 'V-measure', 'Silhouette'],
                       index=['K-means', 'Affinity', 'Spectral', 'Agglomerative'])
print(results)
