import json
import numpy as np
from operator import add

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
usertagsIndex = usertags.zipWithIndex()
all_tags, all_pairs = 0, 0
beta, converge_dist, split_dist = 1.0, 0.001, 0.1
batchSize = 50000
batchId = 0
while (batchId + 1) * batchSize <= numOfDocs:
  # Create batch
  beginId = batchId * batchSize
  endId = (batchId + 1) * batchSize
  docs = usertagsIndex.filter(lambda p: beginId <= p[1] and p[1] < endId) \
                      .map(lambda p: p[0]) \
                      .filter(lambda p: p[0] in busers.value)
  batchId = batchId + 1

  log.info("Current batch: %d" % batchId)

  pairs = docs.flatMap(lambda (u, tags): [(t, u) for t in tags]) \
              .distinct() \
              .cache()

  # Extract tags
  tags = pairs.map(lambda (t, u): t) \
              .distinct() \
              .collect()
  ntags = len(tags)
  btags = sc.broadcast(dict(zip(tags, range(ntags))))
  log.info("(batch %d) number of tags: %d" % (batchId, ntags))

  # Essential matrices
  c_h = pairs.map(lambda (t, u): (t, 1)) \
             .reduceByKey(add) \
             .map(lambda (t, c): (btags.value[t], c)) \
             .sortByKey() \
             .map(lambda l: l[1]).collect()
  c_h = np.array(c_h)

  all_tags = all_tags + c_h.sum()
  all_pairs = all_pairs + pairs.count()

  bp_h = sc.broadcast(1.0 * c_h / all_tags)
  p_uh = pairs.map(lambda (t, u): (t, (busers.value[u], 1.0))) \
              .groupByKey() \
              .map(lambda (t, v): (btags.value[t], get_norm_1d_csr(v, numOfUsers))) \
              .cache()
  p_uh_co_occur = p_uh.map(lambda (h, v): (h, bp_h.value[h] * v)) \
                      .cache()

  if batchId == 1:
    p_t = np.array([0.5, 0.5])
    p_yt_co_occur = kmeans_plus(p_uh, bp_h.value)
  beta, p_t, p_yt_co_occur = search_beta(p_t, p_yt_co_occur, \
      beta, converge_dist, split_dist, bp_h.value, p_uh, p_uh_co_occur)

