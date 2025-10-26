import numpy as np

def hdbscan(X, min_samples=5, min_cluster_size=5):
    from scipy.spatial.distance import pdist, squareform
    D = squareform(pdist(X))
    n = len(X)

    core_dists = np.sort(D, axis=1)[:, min_samples - 1]
    mutual_reachability = np.maximum(D, np.maximum(core_dists[:, None], core_dists[None, :]))

    parent = np.arange(n)
    cluster_labels = -np.ones(n, dtype=int)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    edges = [(i, j, mutual_reachability[i, j]) for i in range(n) for j in range(i + 1, n)]
    edges.sort(key=lambda x: x[2])

    cluster_id = 0
    for i, j, d in edges:
        union(i, j)
        unique_clusters = np.unique([find(x) for x in range(n)])
        for c in unique_clusters:
            members = np.where(np.array([find(x) for x in range(n)]) == c)[0]
            if len(members) >= min_cluster_size and np.all(cluster_labels[members] == -1):
                cluster_labels[members] = cluster_id
                cluster_id += 1

    return cluster_labels
