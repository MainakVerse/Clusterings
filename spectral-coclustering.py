import numpy as np

def spectral_coclustering(X, n_clusters=2):
    from scipy.linalg import svd

    X = np.array(X, dtype=float)
    X_norm = X / np.maximum(X.sum(axis=1, keepdims=True), 1e-10)
    U, S, Vt = svd(X_norm, full_matrices=False)

    features = np.concatenate([U[:, :n_clusters], Vt.T[:, :n_clusters]], axis=0)

    def kmeans(U, k, max_iters=100):
        centroids = U[np.random.choice(len(U), k, replace=False)]
        for _ in range(max_iters):
            labels = np.argmin(np.linalg.norm(U[:, None] - centroids, axis=2), axis=1)
            new_centroids = np.array([U[labels == i].mean(axis=0) for i in range(k)])
            if np.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids
        return labels

    labels = kmeans(features, n_clusters)
    row_labels = labels[:X.shape[0]]
    col_labels = labels[X.shape[0]:]
    return row_labels, col_labels
