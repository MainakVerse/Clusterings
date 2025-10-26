import numpy as np

def fuzzy_c_means(X, c=2, m=2, max_iters=100, tol=1e-5):
    n, d = X.shape
    np.random.seed(42)
    U = np.random.dirichlet(np.ones(c), size=n)

    for _ in range(max_iters):
        U_m = U ** m
        centers = (U_m.T @ X) / U_m.sum(axis=0)[:, None]

        dist = np.zeros((n, c))
        for i in range(c):
            dist[:, i] = np.linalg.norm(X - centers[i], axis=1)
        dist = np.fmax(dist, 1e-10)

        new_U = 1.0 / np.sum((dist[:, :, None] / dist[:, None, :]) ** (2 / (m - 1)), axis=2)

        if np.linalg.norm(new_U - U) < tol:
            break
        U = new_U

    labels = np.argmax(U, axis=1)
    return labels, centers, U
