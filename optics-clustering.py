import numpy as np

def optics(X, eps=1.0, min_samples=5):
    n = len(X)
    reach_dist = np.full(n, np.inf)
    core_dist = np.full(n, np.inf)
    processed = np.zeros(n, dtype=bool)
    order = []

    def neighbors(p):
        return np.where(np.linalg.norm(X - X[p], axis=1) <= eps)[0]

    def core_distance(p, nbrs):
        if len(nbrs) < min_samples:
            return np.inf
        dists = np.sort(np.linalg.norm(X[nbrs] - X[p], axis=1))
        return dists[min_samples - 1]

    def update(nbrs, p, seeds):
        for o in nbrs:
            if processed[o]:
                continue
            new_reach = max(core_dist[p], np.linalg.norm(X[p] - X[o]))
            if np.isnan(reach_dist[o]) or new_reach < reach_dist[o]:
                reach_dist[o] = new_reach
                seeds.append(o)

    for p in range(n):
        if processed[p]:
            continue
        nbrs = neighbors(p)
        processed[p] = True
        order.append(p)
        core_dist[p] = core_distance(p, nbrs)
        if core_dist[p] != np.inf:
            seeds = []
            update(nbrs, p, seeds)
            while seeds:
                q = seeds.pop(np.argmin([reach_dist[s] for s in seeds]))
                nbrs_q = neighbors(q)
                processed[q] = True
                order.append(q)
                core_dist[q] = core_distance(q, nbrs_q)
                if core_dist[q] != np.inf:
                    update(nbrs_q, q, seeds)

    return np.array(order), reach_dist
