import numpy as np

def bisecting_kmeans(X, n_clusters=2, max_iters=100):
    def kmeans(X, k=2, max_iters=100):
        centroids = X[np.random.choice(len(X), k, replace=False)]
        for _ in range(max_iters):
            labels = np.argmin(np.linalg.norm(X[:, None] - centroids, axis=2), axis=1)
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
            if np.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids
        return labels, centroids

    clusters = [X]
    labels_list = [np.zeros(len(X), dtype=int)]

    while len(clusters) < n_clusters:
        largest_idx = np.argmax([len(c) for c in clusters])
        cluster_to_split = clusters.pop(largest_idx)
        base_labels = labels_list.pop(largest_idx)

        labels, _ = kmeans(cluster_to_split, k=2, max_iters=max_iters)
        cluster1 = cluster_to_split[labels == 0]
        cluster2 = cluster_to_split[labels == 1]

        clusters.extend([cluster1, cluster2])
        new_labels = np.zeros(len(X), dtype=int)
        new_labels[base_labels == 0] = 0
        new_labels[base_labels == 1] = 1
        labels_list.extend([new_labels] * 2)

    final_labels = np.zeros(len(X), dtype=int)
    idx = 0
    for cluster in clusters:
        for point in cluster:
            pos = np.where((X == point).all(axis=1))[0][0]
            final_labels[pos] = idx
        idx += 1

    return final_labels
