import numpy as np

def minibatch_kmeans(X, k, batch_size=32, max_iters=100):
    n_samples, _ = X.shape
    centroids = X[np.random.choice(n_samples, k, replace=False)]
    
    for _ in range(max_iters):
        batch_idx = np.random.choice(n_samples, batch_size, replace=False)
        batch = X[batch_idx]
        
        distances = np.linalg.norm(batch[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        for j in range(k):
            points = batch[labels == j]
            if len(points) > 0:
                centroids[j] = centroids[j] + 0.1 * (points.mean(axis=0) - centroids[j])
    
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    final_labels = np.argmin(distances, axis=1)
    return final_labels, centroids
