import numpy as np

def dbscan(X, eps=0.5, min_samples=5):
    n = len(X)
    labels = np.full(n, -1)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    def region_query(point_idx):
        return np.where(np.linalg.norm(X - X[point_idx], axis=1) <= eps)[0]

    def expand_cluster(point_idx, neighbors):
        nonlocal cluster_id
        labels[point_idx] = cluster_id
        i = 0
        while i < len(neighbors):
            n_idx = neighbors[i]
            if not visited[n_idx]:
                visited[n_idx] = True
                n_neighbors = region_query(n_idx)
                if len(n_neighbors) >= min_samples:
                    neighbors = np.unique(np.concatenate((neighbors, n_neighbors)))
            if labels[n_idx] == -1:
                labels[n_idx] = cluster_id
            i += 1

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = region_query(i)
        if len(neighbors) < min_samples:
            labels[i] = -1
        else:
            expand_cluster(i, neighbors)
            cluster_id += 1

    return labels
