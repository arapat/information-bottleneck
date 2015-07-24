import numpy as np

def hartigan_centroids(p_yx, assignments):
    centroids = p_yx.map(lambda (x, v): (assignments[x], (p_x[x], p_x[x] * v))) \
                    .reduceByKey(lambda (a, b): (a[0] + b[0], a[1] + b[1])) \
    return map(lambda p: p[1], centroids)


def hartigan_init(p_yx):
    def partition(v, p1, p2):
        d1 = distance(v, p1)
        d2 = distance(v, p2)
        if d1 < d2:
            return 0
        return 1

    p1 = p_yx.takeSample(False, 1)[1]
    weights = p_yx.map(lambda (x, v): (x, distance(v, p1)) \
                  .sortByKey().collect()
    weights = np.array(weights)**2
    p2_idx = np.random.choice(weights.size, p = weights / weights.sum())
    p2 = p_yx.zipWithIndex() \
             .filter(lambda p: p[1] == p2_idx) \
             .first()[0][1]

    assignments = p_yx.map(lambda (x, v): (x, partition(v, p1, p2))) \
                      .sortByKey() \
                      .map(lambda p: p[1]) \
                      .collect()
    p_yt = hartigan_centroids(p_yx, assignments)
    return assignments, p_yt


def hartigan_twoCentroids(p_x, p_yx):
    assignments, p_yt = hartigan_init(p_yx)
    while True:
        new_assignments = p_yx.map(lambda (x, v): (x, adjust(v, p_yt, assignments[x]))) \
                              .sortByKey() \
                              .map(lambda p: p[1]) \
                              .collect()
        new_assignments = np.array(new_assignments)
        if np.all(assignments == new_assignments):
            break
        assignments = new_assignments
        p_yt = hartigan_centroids(p_yx, assignments)
    return p_yt

