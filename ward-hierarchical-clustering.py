import numpy as np

def ward_hierarchical_clustering(X, n_clusters=2):
    n_samples = len(X)
    clusters = [[i] for i in range(n_samples)]
    centroids = X.copy()
    distances = np.linalg.norm(centroids[:, None] - centroids[None, :], axis=2) ** 2
    np.fill_diagonal(distances, np.inf)

    while len(clusters) > n_clusters:
        i, j = np.unravel_index(np.argmin(distances), distances.shape)
        merged_cluster = clusters[i] + clusters[j]
        merged_centroid = X[merged_cluster].mean(axis=0)

        clusters[i] = merged_cluster
        centroids[i] = merged_centroid
        clusters.pop(j)
        centroids = np.delete(centroids, j, axis=0)
        distances = np.delete(distances, j, axis=0)
        distances = np.delete(distances, j, axis=1)

        for k in range(len(clusters)):
            if k != i:
                diff = centroids[k] - centroids[i]
                distances[i, k] = distances[k, i] = np.sum(diff ** 2)
        distances[i, i] = np.inf

    labels = np.zeros(n_samples, dtype=int)
    for idx, cluster in enumerate(clusters):
        labels[cluster] = idx
    return labels
