from time import time

import json
import numpy as np
from operator import add

def gen_data(batchId, docs, all_tags, all_pairs):
  pairs = docs.flatMap(lambda (u, tags): [(t, u) for t in tags]) \
              .distinct() \
              .cache()

  # Extract tags
  tags = pairs.map(lambda (t, u): t) \
              .distinct() \
              .collect()
  ntags = len(tags)
  btags = sc.broadcast(dict(zip(tags, range(ntags))))
  idtags = dict(zip(range(ntags), tags))
  log.info("(batch %d) number of tags: %d" % (batchId, ntags))

  # Essential matrices
  c_h = pairs.map(lambda (t, u): (t, 1)) \
             .reduceByKey(add) \
             .map(lambda (t, c): (btags.value[t], c)) \
             .sortByKey() \
             .map(lambda l: l[1]).collect()
  c_h = np.array(c_h)

  c_h_sum = c_h.sum()
  pairs_count = pairs.count()
  ratio_tags = 1.0 * all_tags / (all_tags + c_h_sum)
  ratio_pairs = 1.0 * all_pairs / (all_pairs + pairs_count)
  all_tags = all_tags + c_h_sum
  all_pairs = all_pairs + pairs_count

  bp_h = sc.broadcast(1.0 * c_h / all_tags)
  p_uh = pairs.map(lambda (t, u): (t, (busers.value[u], 1.0))) \
              .groupByKey() \
              .map(lambda (t, v): (btags.value[t], get_norm_1d_csr(v, numOfUsers))) \
              .cache()
  p_uh_co_occur = p_uh.map(lambda (h, v): (h, bp_h.value[h] * v)) \
                      .cache()
  return idtags, ratio_tags, ratio_pairs, all_tags, all_pairs, c_h, bp_h, p_uh, p_uh_co_occur


def get_data(batchId, batchSize, usertagsIndex, fromStart = False):
  """
  batchId: start from 0
  """
  beginId = batchId * batchSize
  if fromStart:
    beginId = 0
  endId = (batchId + 1) * batchSize
  return usertagsIndex.filter(lambda p: beginId <= p[1] and p[1] < endId) \
                      .map(lambda p: p[0]) \
                      .filter(lambda p: p[0] in busers.value)


# Parse data
data_url = '/mnt/s3/tweets/elections0[1-9].json,/mnt/s3/tweets/elections1[0-1].json'
data = sc.textFile(data_url) \
         .map(try_load) \
         .filter(lambda x: x)
usertags = data.map(lambda l: (l["user"]["id"], extract_tags(l["text"]))) \
               .cache()

# Assign an index to each user
usersRDD = usertags.map(lambda p: (p[0], set(p[1]))) \
                   .reduceByKey(lambda a, b: a.union(b)) \
                   .map(lambda p: (p[0], len(p[1]))) \
                   .cache()
print "Total number of users:", usersRDD.count()
users = usersRDD.filter(lambda p: p[1] > 1) \
                .map(lambda p: p[0]) \
                .collect()
numOfUsers = len(users)
busers = sc.broadcast(dict(zip(users, range(numOfUsers))))
print "Number of users:", numOfUsers

# Estimate data size
numOfTags = usertags.flatMap(lambda p: set(p[1])).distinct().count()
numOfDocs = usertags.count()
numOfTags, numOfUsers, numOfDocs

# Package tweets to batches
# Generate initial centroids
usertagsIndex = usertags.zipWithIndex()
beta, converge_dist = 128.0, 0.001
batchSize = 25000
sampleBatches = 10
docs = get_data(sampleBatches - 1, batchSize, usertagsIndex, fromStart = True)
idtags, ratio_t, ratio_p, all_tags, all_pairs, c_h, bp_h, p_uh, p_uh_co_occur = \
    gen_data(0, docs, 0, 0)
p_t, p_yt_co_occur, centroid_id = init_centroids(c_h, bp_h.value, p_uh, p_uh_co_occur, beta)

# Streaming...
alpha = 1.0
all_tags, all_pairs = 0, 0
old_data = sc.parallelize([])
batchId = 0
while batchId <= 2 and (batchId + 1) * batchSize <= numOfDocs:
  # Create batch
  timer = time()
  new_data = get_data(batchId, batchSize, usertagsIndex)
  docs = new_data.union(old_data)
  batchId = batchId + 1

  log.info("Current batch: %d" % batchId)

  idtags, ratio_t, ratio_p, all_tags, all_pairs, c_h, bp_h, p_uh, p_uh_co_occur = \
      gen_data(batchId, docs, all_tags, all_pairs)

  p_t, p_yt_co_occur, move, dist = search_beta(p_t, p_yt_co_occur, ratio_t, ratio_p, \
      beta, converge_dist, bp_h.value, p_uh, p_uh_co_occur)

  log.info("batch %d takes %f seconds" % (batchId, time() - timer))

  sample_ratio = alpha * (batchId - 1) / batchId
  old_data = old_data.sample(False, sample_ratio) \
                     .union(new_data.sample(False, 1.0 - sample_ratio)) \
                     .cache()

