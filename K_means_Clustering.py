from sklearn.datasets._samples_generator import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X, _ = make_blobs(n_samples=300, centers=3, cluster_std=2)

plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], s=30, marker='x')
plt.show()

KMeans = KMeans(n_clusters=3)
KMeans.fit(X)
Y = KMeans.predict(X)

plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], c=Y, s=30, marker='x', cmap='plasma')
centers = KMeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.8)
plt.show()