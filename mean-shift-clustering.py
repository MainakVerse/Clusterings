import numpy as np

def mean_shift(X, bandwidth=1.0, max_iters=100):
    X = np.array(X)
    centroids = np.copy(X)

    for _ in range(max_iters):
        new_centroids = []
        for c in centroids:
            distances = np.linalg.norm(X - c, axis=1)
            weights = np.exp(-(distances ** 2) / (2 * (bandwidth ** 2)))
            new_c = np.sum(X * weights[:, np.newaxis], axis=0) / np.sum(weights)
            new_centroids.append(new_c)
        new_centroids = np.array(new_centroids)

        if np.linalg.norm(new_centroids - centroids) < 1e-5:
            break
        centroids = new_centroids

    unique_centroids = []
    for c in centroids:
        if not any(np.linalg.norm(c - uc) < bandwidth / 2 for uc in unique_centroids):
            unique_centroids.append(c)

    return np.array(unique_centroids)
