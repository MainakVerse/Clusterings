import numpy as np

def spectral_biclustering(X, n_clusters=(2, 2)):
    from scipy.linalg import svd

    X = np.array(X, dtype=float)
    X_centered = X - X.mean(axis=0)
    U, S, Vt = svd(X_centered, full_matrices=False)

    n_row_clusters, n_col_clusters = n_clusters
    row_features = U[:, :n_row_clusters]
    col_features = Vt.T[:, :n_col_clusters]

    def kmeans(U, k, max_iters=100):
        centroids = U[np.random.choice(len(U), k, replace=False)]
        for _ in range(max_iters):
            labels = np.argmin(np.linalg.norm(U[:, None] - centroids, axis=2), axis=1)
            new_centroids = np.array([U[labels == i].mean(axis=0) for i in range(k)])
            if np.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids
        return labels

    row_labels = kmeans(row_features, n_row_clusters)
    col_labels = kmeans(col_features, n_col_clusters)
    return row_labels, col_labels
