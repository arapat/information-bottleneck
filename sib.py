from time import time

import numpy as np
from numpy import multiply
from numpy.random import uniform
from scipy.stats import entropy
from scipy.sparse import csr_matrix

def jsd(p, q):
    m = (p + q) / 2.0
    # TODO
    # t1 = np.nan_to_num(multiply(p, np.log2(p / m))).sum()
    # t2 = np.nan_to_num(multiply(q, np.log2(q / m))).sum()
    t1 = np.nan_to_num(p.multiply(np.log2(p / m))).sum()
    t2 = np.nan_to_num(q.multiply(np.log2(q / m))).sum()
    return (t1 + t2) / 2.0

def compute_jsd(b, p_vc):
    return np.array([jsd(b, p_vc[c, :]) for c in range(p_vc.shape[0])])

def get_free_energy(p_cn_trans, p_vc, p_c, js_div, beta):
    """
      Free energy: F = <D> - H/\beta, specifically
      sum_n( p(n) sum_c (p(c|n)JSD(n, c)) ) + sum_{n, c} p(c|n) log( p(n|c) ) / \beta
    """
    def compute_p_nc():
        p = p_cn_trans.multiply(p_n) / p_c.toarray()[0][None, :]
        return csr_matrix(p)
  
    def get_entropy(p_nc):
        # return -np.sum(p_cn_trans * np.log2(p_nc))
        ent = np.log2(p_nc.toarray())
        temp = np.sum(p_nc.multiply(ent), axis=1)
        return -(p_c.multiply(temp)).sum()
        
    def get_distortion():
        return js_div.map(lambda (a, values): p_n[0, a] * \
                p_cn_trans[:, a].multiply(values).sum()) \
                     .sum()
    
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
    return csr_matrix(result)


def evaluate(js_div, pc, beta):
    dists = pc.multiply(np.exp(-beta * js_div))
    return csr_matrix(dists) / dists.sum()

def converge(p_cn_trans, beta, convergeDist, p_n, p_vn, p_vn_co_occur):
    def outer(p1, p2):
        p = np.outer(p1.toarray(), p2.toarray())
        return csr_matrix(p)

    p_vc = None
    iterations = 0
    
    # To compute free energy
    pc = None
    js_div = None

    while True:
        iterations = iterations + 1

        # p(c)
        pc = p_cn_trans.dot(p_n.T) # pn[n] * p_cn[n][c]

        # p(v|c)
        # computed as: p_cn[n][c] * p_vn[n][v] * pn[n] / pc[c]
        p_vc = p_vn_co_occur.map(lambda (a, b): outer(p_cn_trans[:, a], b[0, :])) \
                            .sum() / pc.toarray()[0][:, None]
        p_vc = csr_matrix(p_vc)

        # new p(c|n)
        # computed as: pc[c] * exp(-beta * jsd(b, p_vc[c]))
        if js_div:
            js_div.unpersist()
        js_div = p_vn.map(lambda (a, b): (a, compute_jsd(b, p_vc))).cache()
        print js_div.take(1) # TODO
        new_p_cn = js_div.map(lambda (a, values): (a, evaluate(values, pc, beta))) \
                         .sortByKey() \
                         .collect()
        # TODO
        # special case -> p_cn[n][k] = 0.0

        max_diff = 0.0
        k = 0
        for n in range(len(nouns)):
            if max_diff <= convergeDist and new_p_cn[k][0] == n:
                diff = jsd(p_cn_trans[:, n], new_p_cn[k][1])
                max_diff = max(diff, max_diff)
                k = k + 1
        if max_diff <= convergeDist:
            break

        p_cn_trans = csr_matrix(new_p_cn.T)
    
    free_energy = get_free_energy(p_cn_trans, p_vc, pc, js_div, beta)
    js_div.unpersist()

    return p_cn_trans, p_vc, free_energy, iterations


def distributional_clustering(p_n, p_vn, p_vn_co_occur, split_threshold, beta, convergeDist, splitDist, alpha):
    TRIAL = 2
    # To return
    p_cn_trans = []
    child = []
    is_leaf = []
    free_energy = []
    split_point = []
    # Intermediate results
    np_vc = []
    new_p_cn_trans = []
    
    def append(distr):
        p_cn_trans.append(distr)
        child.append(())
        is_leaf.append(True)
        return len(is_leaf) - 1
    
    def append_prt(i1, i2, prt):
        is_leaf[prt] = False
        p_cn_trans[prt] = np.zeros(len(nouns))
        child[prt] = (i1, i2)

    def diverge_detect(split, beta, left, right):
        # diverge detection
        js_distance = [jsd(np_vc[k], np_vc[k + 1]) for k in range(0, len(np_vc) - 1, 2)]
        hit = np.max(js_distance) > splitDist
        updated = hit and left + 1.0 >= beta
        if updated:
            leaves = [i for i in range(len(is_leaf)) if is_leaf[i]]
            for k, i, p1, p2 in zip(range(len(js_distance)), leaves, \
                    range(0, len(np_vc), 2), range(1, len(np_vc), 2)):
                if js_distance[k] > splitDist:
                    i1, i2 = append(new_p_cn_trans[p1]), append(new_p_cn_trans[p2])
                    append_prt(i1, i2, i)
                    split = split + 1
                else:
                    p_cn_trans[i] = new_p_cn_trans[p1] + new_p_cn_trans[p2]

        # log
        if updated:
            split_point.append(beta)
            log.info('===== beta %f updated =====' % (beta))
        elif hit:
            log.info('beta = %f, left = %f, right = %f successed. Closer the gap.' % (beta, left, right))
            
        # for next iteration
        if updated:
            left = beta
            right = np.inf
        elif hit:
            right = beta
            beta = (left + right) / 2.0
        else:
            left = beta
            beta = np.min([beta * 2.0, (left + right) / 2.0])
        return hit, updated, split, beta, left, right

    # root initialization
    init_p_cn_trans = csr_matrix(np.ones(len(nouns)))
    init_cn, init_vc, fe, iterations = converge(init_p_cn_trans, beta, convergeDist, \
            p_n, p_vn, p_vn_co_occur)
    append(init_p_cn_trans)
    free_energy.append((beta, fe))
    split_point.append(beta)
    # print 'initial (max) entropy', entropy(get_preference(distr, beta)[0])

    log.info('root completed.')

    left = beta
    right = np.inf
    split = 0
    trial_count = 1
    while split < split_threshold:
        log.info("trial %d with beta %f" % (trial_count, beta))
        
        # pertubate and converge
        adjusted_p_cn_trans = pertubate(p_cn_trans, is_leaf, alpha)
        timer = time()
        new_p_cn_trans, np_vc, fe, iterations = converge(adjusted_p_cn_trans, beta, convergeDist, p_n, p_vn, p_vn_co_occur)
        log.info('Converge time %f seconds (%d iterations)' % (time() - timer, iterations))

        free_energy.append((beta, fe))

        # diverge detection
        diverged, updated, split, beta, left, right = diverge_detect(split, beta, left, right)
        if updated and alpha >= 1e-5:
            alpha = alpha * 0.01
        if diverged or trial_count >= TRIAL:
            trial_count = 0
        trial_count = trial_count + 1

    free_energy.sort()
    return p_cn_trans, child, is_leaf, free_energy, split_point, beta

