import numpy as np

def k_medoids(X, k, max_iters=100):
    n = len(X)
    medoids = np.random.choice(n, k, replace=False)
    labels = np.zeros(n, dtype=int)

    for _ in range(max_iters):
        distances = np.linalg.norm(X[:, None] - X[medoids], axis=2)
        labels = np.argmin(distances, axis=1)

        new_medoids = np.copy(medoids)
        for i in range(k):
            cluster_points = np.where(labels == i)[0]
            if len(cluster_points) == 0:
                continue
            intra_dists = np.sum(np.linalg.norm(X[cluster_points][:, None] - X[cluster_points], axis=2), axis=1)
            best_point = cluster_points[np.argmin(intra_dists)]
            new_medoids[i] = best_point

        if np.all(medoids == new_medoids):
            break
        medoids = new_medoids

    return labels, X[medoids]
