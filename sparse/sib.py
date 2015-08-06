import numpy as np

def hartigan_adjust(weight, vector, p_yt, c):
    assert(len(p_yt) == 2)
    dist = np.inf
    best = c
    for i in range(len(p_yt)):
        if i == c:
            centroid = (p_yt[i][1] - vector) / (p_yt[i][0] - weight)
        else:
            centroid = p_yt[i][1] / p_yt[i][0]
        n_dist = distance(centroid, vector, True, True)
        if n_dist < dist:
            dist = n_dist
            best = i
    return best


def hartigan_centroids(p_yx, p_x, assignments):
    centroids = p_yx.map(lambda (x, v): (assignments[x], (p_x[x], p_x[x] * v.toarray()[0]))) \
                    .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1])) \
                    .collect()
    return map(lambda p: p[1], centroids)


def hartigan_init(p_yx, p_x):
    def partition(v, p1, p2):
        d = distance(v.toarray()[0], np.array([p1, p2]))
        assert(d.shape == (2,))
        if d[0] < d[1]:
            return 0
        return 1

    p1 = p_yx.takeSample(False, 1)[0][1].toarray()[0]
    weights = p_yx.map(lambda (x, v): (x, distance(v.toarray()[0], p1, True, True))) \
                  .sortByKey().collect()[0]
    weights = np.array(weights)**2
    p2_idx = np.random.choice(weights.size, p = weights / weights.sum())
    p2 = p_yx.zipWithIndex() \
             .filter(lambda p: p[1] == p2_idx) \
             .first()[0][1].toarray()[0]

    assignments = p_yx.map(lambda (x, v): (x, partition(v, p1, p2))) \
                      .sortByKey() \
                      .map(lambda p: p[1]) \
                      .collect()
    p_yt = hartigan_centroids(p_yx, p_x, assignments)
    return assignments, p_yt


def hartigan_twoCentroids(p_x, p_yx, numOfX):
    assignments, p_yt = hartigan_init(p_yx, p_x)
    loop = 0
    while True:
        loop = loop + 1
        log.info("iteration " + str(loop))
        new_assignments = p_yx.map(lambda (x, v): \
                (x, hartigan_adjust(p_x[x], v.toarray()[0], p_yt, assignments[x]))) \
                              .sortByKey() \
                              .map(lambda p: p[1]) \
                              .collect()
        log.info("assignments updated")
        new_assignments = np.array(new_assignments)
        if np.all(assignments == new_assignments):
            break
        assignments = new_assignments
        p_yt = hartigan_centroids(p_yx, p_x, assignments)

    p_tx = np.zeros((numOfX, 2))
    for k in range(numOfX):
        p_tx[k][assignments[k]] = 1.0
    return p_tx, assignments

