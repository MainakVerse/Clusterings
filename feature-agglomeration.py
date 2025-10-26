import numpy as np

def feature_agglomeration(X, n_clusters=2):
    n_features = X.shape[1]
    clusters = [[i] for i in range(n_features)]
    distances = np.zeros((n_features, n_features))

    for i in range(n_features):
        for j in range(i + 1, n_features):
            distances[i, j] = np.corrcoef(X[:, i], X[:, j])[0, 1]
            distances[j, i] = distances[i, j]

    distances = 1 - np.abs(distances)
    np.fill_diagonal(distances, np.inf)

    while len(clusters) > n_clusters:
        i, j = np.unravel_index(np.argmin(distances), distances.shape)
        clusters[i].extend(clusters[j])
        clusters.pop(j)
        distances = np.delete(distances, j, axis=0)
        distances = np.delete(distances, j, axis=1)
        for k in range(len(clusters)):
            if k != i:
                corr = np.mean([
                    np.corrcoef(X[:, f1], X[:, f2])[0, 1]
                    for f1 in clusters[i]
                    for f2 in clusters[k]
                ])
                distances[i, k] = distances[k, i] = 1 - np.abs(corr)
        distances[i, i] = np.inf

    reduced = np.zeros((X.shape[0], len(clusters)))
    for idx, cluster in enumerate(clusters):
        reduced[:, idx] = X[:, cluster].mean(axis=1)
    return reduced
