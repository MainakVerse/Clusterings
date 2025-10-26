import numpy as np

def self_organizing_map(X, grid=(5, 5), lr=0.5, sigma=None, max_iters=100):
    n, d = X.shape
    rows, cols = grid
    if sigma is None:
        sigma = max(rows, cols) / 2
    weights = np.random.rand(rows, cols, d)

    def neighborhood(center, sigma):
        r, c = np.indices((rows, cols))
        dist_sq = (r - center[0])**2 + (c - center[1])**2
        return np.exp(-dist_sq / (2 * sigma**2))

    for it in range(max_iters):
        x = X[np.random.randint(0, n)]
        dists = np.linalg.norm(weights - x, axis=2)
        bmu_idx = np.unravel_index(np.argmin(dists), (rows, cols))
        h = neighborhood(bmu_idx, sigma)
        weights += lr * h[..., None] * (x - weights)

        lr *= 0.99
        sigma *= 0.99

    return weights
