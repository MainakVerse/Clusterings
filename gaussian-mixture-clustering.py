import numpy as np

def gaussian_mixture(X, n_components=2, max_iters=100, tol=1e-6):
    n_samples, n_features = X.shape
    np.random.seed(42)
    means = X[np.random.choice(n_samples, n_components, replace=False)]
    covariances = [np.cov(X, rowvar=False)] * n_components
    weights = np.ones(n_components) / n_components

    def gaussian(x, mean, cov):
        inv_cov = np.linalg.inv(cov)
        det_cov = np.linalg.det(cov)
        norm = 1 / np.sqrt((2 * np.pi) ** n_features * det_cov)
        diff = x - mean
        return norm * np.exp(-0.5 * np.sum(diff @ inv_cov * diff, axis=1))

    for _ in range(max_iters):
        resp = np.zeros((n_samples, n_components))
        for k in range(n_components):
            resp[:, k] = weights[k] * gaussian(X, means[k], covariances[k])
        resp /= resp.sum(axis=1, keepdims=True)

        N_k = resp.sum(axis=0)
        new_means = (resp.T @ X) / N_k[:, np.newaxis]
        new_covariances = []
        for k in range(n_components):
            diff = X - new_means[k]
            new_cov = (resp[:, k][:, np.newaxis] * diff).T @ diff / N_k[k]
            new_covariances.append(new_cov + 1e-6 * np.eye(n_features))
        new_weights = N_k / n_samples

        if np.allclose(means, new_means, atol=tol):
            break
        means, covariances, weights = new_means, new_covariances, new_weights

    labels = np.argmax(resp, axis=1)
    return labels, means, covariances, weights
