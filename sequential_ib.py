import sys
import urllib

from operator import itemgetter

import numpy as np

# AWS credentials
ACCESS_KEY = "" # ACCESS_KEY
SECRET_KEY = "" # SECRET_KEY
ENCODED_SECRET_KEY = urllib.quote(SECRET_KEY, "")
AWS_BUCKET_NAME = "" # AWS bucket name
MOUNT_NAME = "s3"

# logs
log = sys.stderr
# log = open("sib_multi.log", "w")

# Mount S3 to local
dbutils.fs.unmount("/mnt/%s" % MOUNT_NAME)
dbutils.fs.mount("s3n://%s:%s@%s" % (ACCESS_KEY, ENCODED_SECRET_KEY, AWS_BUCKET_NAME), "/mnt/%s" % MOUNT_NAME)

# Preprocess data
data = sc.textFile("/mnt/s3/train")
data.take(2)

pairs = data.map(lambda p: p.strip().split('\t')) \
            .map(lambda p: (p[1].strip(), (p[2].strip(), int(p[0])))) # ('noun', ('verb', count))

nouns = sorted(pairs.map(lambda p: p[0]).distinct().collect())
verbs = sorted(pairs.map(lambda p: p[1][0]).distinct().collect())
total = pairs.map(lambda p: p[1][1]).sum()

c_vn = pairs.groupByKey().map(lambda (a, b): (a, sorted(b))) # ('noun', [('verb1', count1), ('verb2', count2)])
n_sum = c_vn.map(lambda (a, b): (a, sum_of_tuples(b, itemgetter(1))))
raw_p_vn = c_vn.join(n_sum).map(lambda (a, b): (a, normalize(b[0], b[1]))) # ('noun', [('verb1', p1), ('verb2', p2)])
raw_p_n = n_sum.map(lambda (a, b): (a, 1.0 * b / total)).sortByKey().collect()

# Vectors that will be used later
p_n = vectorize(raw_p_n, nouns)
p_vn = raw_p_vn.map(lambda (a, b): (nouns.index(a), vectorize(b, verbs))) \
               .cache()
p_vn_co_occur = p_vn.map(lambda (a, b): (a, b * p_n[a])) \
                 .cache()

# Entropy
p_entropy, result, min_beta = result_entropy(p_n, p_vn, p_vn_co_occur, 1.0, 1, 10)
hard_clusters = classify(result)

hard_clusters.sort(key = lambda v: len(v), reverse=True)

filter_rule = lambda a: get_group_id(a, hard_clusters) == 0
p_vn_l = p_vn.filter(lambda (a, b): filter_rule(a)) \
               .cache()
p_vn_co_occur_l = p_vn_co_occur.filter(lambda (a, b): filter_rule(a)) \
                               .cache()
p_n_l = np.array([p_n[k] * filter_rule(k) for k in range(len(nouns))])
p_n_l = p_n_l / p_n_l.sum()
