from time import time

import numpy as np
from numpy.random import uniform
from scipy.stats import entropy

def jsd(p, q):
    m = (p + q) / 2.0
    t1 = np.nan_to_num(p * np.log2(p / m)).sum()
    t2 = np.nan_to_num(q * np.log2(q / m)).sum()
    return (t1 + t2) / 2.0

def kld(p, q):
    # TODO: NOT used.
    # TODO: bug
    # in case when p[k] > 0 but q[k] = 0
    result = np.sum([a * np.log2(a / b) for a, b in zip(p, q) if a > 1e-30])
    return result

def compute_jsd(b, p_vc):
    return np.array([jsd(b, p_vc[c]) for c in range(len(p_vc))])

def get_free_energy(p_cn_trans, p_vc, p_c, js_div, beta):
    """
      Free energy: F = <D> - H/\beta, specifically
      sum_n( p(n) sum_c (p(c|n)JSD(n, c)) ) + sum_{n, c} p(c|n) log( p(n|c) ) / \beta
    """
    N, C = len(p_n), len(p_c)
    def compute_p_nc():
        return p_cn_trans * np.hstack(p_n) / np.vstack(p_c)
  
    def get_entropy(p_nc):
        # return -np.sum(p_cn_trans * np.log2(p_nc))
        return -(p_c * np.sum(p_nc * np.log2(p_nc), axis=1)).sum()
        
    def get_distortion():
        return js_div.map(lambda (a, values): p_n[a] * np.sum(p_cn_trans.T[a] * values)).sum()
    
    # TODO
    distortion = get_distortion()
    entropy = get_entropy(compute_p_nc())
    return distortion, entropy, entropy / beta

def pertubate(p_cn_trans, is_leaf, alpha):
    result = []
    for c in range(len(is_leaf)):
        if is_leaf[c]:
            eps = alpha * uniform(-0.5, 0.5, len(nouns))
            result.append(p_cn_trans[c] * (0.5 + eps))
            result.append(p_cn_trans[c] * (0.5 - eps))
    return np.array(result)


def evaluate(js_div, pc, beta):
    dists = pc * np.exp(-beta * js_div)
    return dists / dists.sum()

def converge(p_cn_trans, beta, convergeDist, p_n, p_vn, p_vn_co_occur, log=False):
    def merge(tuples, array):
        result = []
        t = 0
        for k in range(0, len(array)):
            if t < len(tuples) and tuples[t][0] == k:
                result.append(tuples[t][1])
                t = t + 1
            else:
                result.append(array[k])
        return result

    K = p_cn_trans.shape[0]
    p_vc = None
    iterations = 0
    
    # To compute free energy
    pc = None
    js_div = None

    while True:
        iterations = iterations + 1
        p_cn = p_cn_trans.T

        # p(c)
        pc = p_cn_trans.dot(p_n) # pn[n] * p_cn[n][c]

        # p(v|c)
        # computed as: p_cn[n][c] * p_vn[n][v] * pn[n] / pc[c]
        p_vc = p_vn_co_occur.map(lambda (a, b): np.outer(p_cn[a], b)) \
                            .sum() / np.vstack(pc)

        # new p(c|n)
        # computed as: pc[c] * exp(-beta * jsd(b, p_vc[c]))
        if js_div:
            js_div.unpersist()
        js_div = p_vn.map(lambda (a, b): (a, compute_jsd(b, p_vc))).cache()
        new_p_cn = js_div.map(lambda (a, values): (a, evaluate(values, pc, beta))) \
                         .sortByKey() \
                         .collect()
        new_p_cn = np.array(merge(new_p_cn, p_cn))

        max_diff = 0.0
        for n in range(len(nouns)):
            if max_diff <= convergeDist:
                diff = jsd(p_cn[n], new_p_cn[n])
                max_diff = max(diff, max_diff)
        if max_diff <= convergeDist:
            break

        p_cn_trans = new_p_cn.T
    
    free_energy = get_free_energy(p_cn_trans, p_vc, pc, js_div, beta)
    js_div.unpersist()

    return p_cn_trans, p_vc, free_energy, iterations


def distributional_clustering(p_n, p_vn, p_vn_co_occur, split_threshold, beta, delta, convergeDist, splitDist, alpha):
    TRIAL = 2
    # To return
    p_cn_trans = []
    child = []
    is_leaf = []
    free_energy = []
    split_point = []
    # Intermediate results
    split = 0
    left = beta
    right = np.inf
    np_vc = []
    new_p_cn_trans = []
    
    def append(distr):
        p_cn_trans.append(distr)
        child.append(())
        is_leaf.append(True)
        return len(is_leaf) - 1
    
    def append_prt(i1, i2, prt):
        is_leaf[prt] = False
        child[prt] = (i1, i2)
        is_updated[prt] = True

    def diverge_detect():
        # diverge detection
        js_distance = [jsd(np_vc[k], np_vc[k + 1]) for k in range(0, len(np_vc) - 1, 2)]
        hit = np.max(js_distance) > splitDist
        updated = hit and left + 1.0 >= beta
        if updated:
            leaves = [i for i in range(len(is_leaf)) if is_leaf[i]]
            for i, p1, p2 in zip(leaves, range(0, len(np_vc), 2), range(1, len(np_vc), 2)):
                if js_distance[k] > splitDist:
                    i1, i2 = append(new_p_cn_trans[p1]), append(new_p_cn_trans[p2])
                    append_prt(i1, i2, i)
                    split = split + 1
                else:
                    p_cn_trans[i] = new_p_cn_trans[p1] + new_p_cn_trans[p2]

        # log
        if updated:
            split_point.append(beta)
            log.write('===== beta %f distance between centroids %f =====' % (beta, diff) + '\n')
        elif hit:
            log.write('beta = %f, left = %f, right = %f successed. Closer the gap.' % (beta, left, right) + '\n')
            
        # for next iteration
        if updated:
            left = beta
            right = np.inf
            # TODO: How to decide alpha?
            if alpha >= 1e-5:
              alpha = alpha * 0.01
        elif hit:
            right = beta
            beta = (left + right) / 2.0
        else:
            left = beta
            beta = np.min(beta * 2.0, (left + right) / 2.0)
        return hit

    # root initialization
    init_cn, init_vc, fe, iterations = converge(np.array([[1.0] * len(nouns)]), beta, convergeDist, p_n, p_vn, p_vn_co_occur)
    append([1.0] * len(nouns))
    free_energy.append((beta, fe))
    split_point.append(beta)
    # print 'initial (max) entropy', entropy(get_preference(distr, beta)[0])

    log.write('root completed.' + '\n')
    log.write("------------------------" + '\n')

    trial_count = 1
    while split < split_threshold:
        log.write("trial %d with beta %f" % (trial_count, beta) + '\n')
        
        # pertubate and converge
        adjusted_p_cn_trans = pertubate(p_cn_trans, is_leaf, alpha)
        timer = time()
        new_p_cn_trans, np_vc, fe, iterations = converge(adjusted_p_cn_trans, beta, convergeDist, p_n, p_vn, p_vn_co_occur)
        log.write('Converge time %f seconds (%d iterations)\n' % (time() - timer, iterations))

        free_energy.append((beta, fe))

        # diverge detection
        diverged = diverge_detect()
        if diverged or trial_count >= TRIAL:
            trial_count = 0
        trial_count = trial_count + 1

        log.write("------------------------" + '\n')

    free_energy.sort()
    return p_cn_trans, child, is_leaf, free_energy, split_point, beta

