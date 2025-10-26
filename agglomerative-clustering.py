import numpy as np

def agglomerative_clustering(X, n_clusters=2):
    n_samples = len(X)
    clusters = [[i] for i in range(n_samples)]
    distances = np.linalg.norm(X[:, None] - X[None, :], axis=2)
    np.fill_diagonal(distances, np.inf)

    while len(clusters) > n_clusters:
        i, j = np.unravel_index(np.argmin(distances), distances.shape)
        clusters[i].extend(clusters[j])
        clusters.pop(j)
        distances = np.delete(distances, j, axis=0)
        distances = np.delete(distances, j, axis=1)
        for k in range(len(clusters)):
            if k != i:
                dist = np.mean(np.linalg.norm(X[clusters[i]] - X[clusters[k]][:, None], axis=2))
                distances[i, k] = distances[k, i] = dist
        distances[i, i] = np.inf

    labels = np.zeros(n_samples, dtype=int)
    for idx, cluster in enumerate(clusters):
        labels[cluster] = idx
    return labels
