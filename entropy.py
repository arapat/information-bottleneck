# Helper functions for computing entropy
import numpy as np

from scipy.stats import entropy

def compute_entropy(p):
    total = len(p)
    stats = []
    values = set(p)
    for v in values:
        stats.append(1.0 * p.count(v) / total)
    return entropy(stats, base = 2.0)

def hard_clustering(p_cn_trans, is_leaf):
    result = [None] * len(nouns)
    for c in range(len(is_leaf)):
      if is_leaf[c]:
        for n in range(len(nouns)):
          if not result[n] or result[n][1] < p_cn_trans[c][n]:
            result[n] = (c, p_cn_trans[c][n])
    return [p[0] for p in result]

def result_entropy(p_n, p_vn, p_vn_co_occur, min_beta, K, trials = 10):
    p_entropy = [0.0]
    result = ['' for i in range(len(nouns))]
    min_beta = np.inf
    loop = 0
    while loop < trials:
        p_cn_trans, child, is_leaf, free_energy, split_point, beta = distributional_clustering(p_n, p_vn, p_vn_co_occur, K, \
                                        beta = min_beta, delta = 0.5, convergeDist = 0.001, splitDist = 0.05, alpha = 0.1)
        for k, c in zip(range(len(nouns)), hard_clustering(p_cn_trans, is_leaf)):
            result[k] = result[k] + '(' + str(c) + ')'
        p_entropy.append(compute_entropy(result))
        min_beta = min(min_beta, beta)
        loop = loop + 1
        log.write("LOOP: %d" % loop + '\n')

    return p_entropy, result, beta
  
def classify(signatures):
    clusters = {}
    items = []
    for n, s in enumerate(signatures):
        if s not in clusters:
            clusters[s] = len(items)
            items.append([])
        items[clusters[s]].append(n)
    return [item for item in items]

def get_group_id(n, hard_clusters):
    for gid, v in enumerate(hard_clusters):
        if n in v:
            return gid
    return -1

