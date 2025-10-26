import numpy as np

def spectral_clustering(X, k=2):
    from scipy.spatial.distance import pdist, squareform
    from scipy.linalg import eigh

    W = np.exp(-squareform(pdist(X)) ** 2 / (2 * np.median(pdist(X)) ** 2))
    np.fill_diagonal(W, 0)
    D = np.diag(W.sum(axis=1))
    L = D - W

    eigvals, eigvecs = eigh(L, D)
    H = eigvecs[:, :k]
    H /= np.linalg.norm(H, axis=1, keepdims=True)

    def kmeans(U, k, max_iters=100):
        centroids = U[np.random.choice(len(U), k, replace=False)]
        for _ in range(max_iters):
            labels = np.argmin(np.linalg.norm(U[:, None] - centroids, axis=2), axis=1)
            new_centroids = np.array([U[labels == j].mean(axis=0) for j in range(k)])
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
        return labels

    return kmeans(H, k)
