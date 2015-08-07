import numpy as np
from operator import add

data_url = '/mnt/s3/train'
data = sc.textFile(data_url)

pairs = data.map(lambda p: p.strip().split('\t')) \
            .map(lambda p: (p[1].strip(), (p[2].strip(), int(p[0])))) # ('noun', ('verb', count))

nouns = sorted(pairs.map(lambda p: p[0]).distinct().collect())
verbs = sorted(pairs.map(lambda p: p[1][0]).distinct().collect())
numOfNouns = len(nouns)
numOfVerbs = len(verbs)

bnouns = sc.broadcast(dict(zip(nouns, range(len(nouns)))))
bverbs = sc.broadcast(dict(zip(verbs, range(len(verbs)))))

c_n = pairs.map(lambda (n, (v, c)): (n, c)) \
           .reduceByKey(add) \
           .map(lambda (n, c): (bnouns.value[n], c)) \
           .sortByKey() \
           .map(lambda l: l[1]).collect()
c_n = np.array(c_n)

# Essential matrices
numOfN = len(c_n)
bp_n = sc.broadcast(1.0 * c_n / c_n.sum())
p_vn = pairs.map(lambda (n, (v, c)): (bnouns.value[n], (bverbs.value[v], c))) \
            .groupByKey() \
            .map(lambda (n, v): (n, get_norm_1d_csr(v, numOfVerbs))) \
            .cache()
p_vn_co_occur = p_vn.map(lambda (n, v): (n, bp_n.value[n] * v)) \
                    .cache()

traces, entropy = split_entropy(1.0, 0.001, 0.1, numOfN, bp_n.value, p_vn, p_vn_co_occur)

