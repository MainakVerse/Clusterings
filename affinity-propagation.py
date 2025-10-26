import numpy as np

def affinity_propagation(X, damping=0.9, max_iters=200, preference=None):
    n = len(X)
    S = -np.square(np.linalg.norm(X[:, None] - X[None, :], axis=2))
    if preference is None:
        preference = np.median(S)
    np.fill_diagonal(S, preference)

    R = np.zeros((n, n))
    A = np.zeros((n, n))

    for _ in range(max_iters):
        AS = A + S
        R_new = S - AS.max(axis=1, keepdims=True)
        R_new[np.arange(n), AS.argmax(axis=1)] = S[np.arange(n), AS.argmax(axis=1)] - np.partition(AS, -2, axis=1)[:, -2]
        R = damping * R + (1 - damping) * R_new

        Rp = np.maximum(R, 0)
        np.fill_diagonal(Rp, R.diagonal())
        A_new = np.sum(Rp, axis=0, keepdims=True) - Rp
        A_new = np.minimum(0, A_new)
        np.fill_diagonal(A_new, np.sum(Rp, axis=0) - Rp.diagonal())
        A = damping * A + (1 - damping) * A_new

    E = A + R
    exemplars = np.where(np.diag(E) > 0)[0]
    labels = np.array([np.argmin(np.linalg.norm(X - X[e], axis=1)) for e in exemplars])
    return labels
