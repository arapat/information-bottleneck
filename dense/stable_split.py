import numpy as np

def init_free_energy(p_n, p_vn, p_vn_co_occur):
    convergeDist = 0.001
    beta = 1.0
    init_cn, init_vc, fe, iterations = converge(np.array([[1.0] * len(nouns)]), beta, convergeDist, p_n, p_vn, p_vn_co_occur)
    return fe
    
def single_stable_split(p_n, p_vn, p_vn_co_occur, p_cn_trans, beta, convergeDist, splitDist, alpha):
    # helper
    is_leaf = [True]
    # Intermediate results
    np_vc = []
    new_p_cn_trans = []
    
    def diverge_detect(split, beta):
        # diverge detection
        if js_distance > splitDist:
            p_cn_trans.append(new_p_cn_trans[0])
            p_cn_trans.append(new_p_cn_trans[1])
            return True
        return False

    free_energy = None
    trial_count = 0
    split = False
    while not split:
        log.info("trial %d with beta %f" % (trial_count, beta))
        
        # pertubate and converge
        adjusted_p_cn_trans = pertubate(p_cn_trans, is_leaf, alpha)
        timer = time()
        new_p_cn_trans, np_vc, fe, iterations = converge(adjusted_p_cn_trans, beta, convergeDist, p_n, p_vn, p_vn_co_occur)
        log.info('Converge time %f seconds (%d iterations)' % (time() - timer, iterations))

        free_energy = fe

        # diverge detection
        js_distance = jsd(np_vc[0], np_vc[1])
        if js_distance > splitDist:
            p_cn_trans[0] = np.zeros(len(p_cn_trans[0]))
            p_cn_trans.append(new_p_cn_trans[0])
            p_cn_trans.append(new_p_cn_trans[1])
            split = True
        trial_count = trial_count + 1

    return p_cn_trans, free_energy, trial_count

def stable_split_entropy(p_n, p_vn, p_vn_co_occur, p_cn_trans, beta, alpha, trials = 10):
    free_energy = []
    num_of_trials = []
    p_entropy = [0.0]
    result = ['' for i in range(len(nouns))]
    loop = 0
    while loop < trials:
        backup = p_cn_trans[:]
        p_cn_trans, fe, trial_count = single_stable_split(p_n, p_vn, p_vn_co_occur, \
            p_cn_trans, beta = beta, convergeDist = 0.001, splitDist = 0.05, alpha = alpha)
        for k, c in zip(range(len(nouns)), hard_clustering(p_cn_trans)):
            result[k] = result[k] + '(' + str(c) + ')'
        p_entropy.append(compute_entropy(result))
        free_energy.append(fe)
        num_of_trials.append(trial_count)
        
        p_cn_trans = backup
        loop = loop + 1
        log.info("LOOP: %d" % loop)
    return p_entropy, result, free_energy, num_of_trials

