import numpy as np

def birch(X, threshold=0.5, branching_factor=50, n_clusters=3, max_iters=100):
    class CFNode:
        def __init__(self, centroid, n_points=1):
            self.centroid = centroid
            self.n_points = n_points

    def merge(node, x):
        new_n = node.n_points + 1
        node.centroid = (node.centroid * node.n_points + x) / new_n
        node.n_points = new_n
        return node

    clusters = [CFNode(X[0])]
    for x in X[1:]:
        distances = [np.linalg.norm(x - node.centroid) for node in clusters]
        idx = np.argmin(distances)
        if distances[idx] < threshold:
            merge(clusters[idx], x)
        elif len(clusters) < branching_factor:
            clusters.append(CFNode(x))
        else:
            merge(clusters[idx], x)

    centroids = np.array([node.centroid for node in clusters])

    def kmeans(U, k, max_iters=100):
        centroids = U[np.random.choice(len(U), k, replace=False)]
        for _ in range(max_iters):
            labels = np.argmin(np.linalg.norm(U[:, None] - centroids, axis=2), axis=1)
            new_centroids = np.array([U[labels == j].mean(axis=0) for j in range(k)])
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
        return labels, centroids

    labels, final_centroids = kmeans(centroids, n_clusters)
    return labels, final_centroids
