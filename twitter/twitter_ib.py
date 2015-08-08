import json
import numpy as np
from operator import add

data_url = '/mnt/s3/tweets/elections[01-10].json'
data = sc.textFile(data_url) \
         .filter(json_validate)

# Parse data
jsonObj = data.map(lambda l: json.loads(l))
usertags = jsonObj.map(lambda l: (l["user"]["id"], extract_tags(l["text"]))) \
                  .flatMap(lambda (u, tags): [(t, u) for t in tags]) \
                  .distinct() \
                  .cache()

# Extract tags
tags = usertags.map(lambda (t, u): (t, 1)) \
               .reduceByKey(add) \
               .sortBy(lambda p: p[1], ascending=False) \
               .map(lambda p: p[0]) \
               .collect()
numOfTags = len(tags)
btags = sc.broadcast(dict(zip(tags, range(numOfTags))))

# Extract users
users = usertags.map(lambda (t, u): (u, 1)) \
                .reduceByKey(add) \
                .sortBy(lambda p: p[1], ascending=False) \
                .map(lambda p: p[0]) \
                .collect()
numOfUsers = len(users)
busers = sc.broadcast(dict(zip(users, range(numOfUsers))))

# Essential matrices
c_h = usertags.map(lambda (t, u): (t, 1)) \
              .reduceByKey(add) \
              .map(lambda (t, c): (btags.value[t], c)) \
              .sortByKey() \
              .map(lambda l: l[1]).collect()
c_h = np.array(c_h)
bp_h = sc.broadcast(1.0 * c_h / c_h.sum())
p_uh = usertags.map(lambda (t, u): (t, (busers.value[u], 1.0))) \
               .groupByKey() \
               .map(lambda (t, v): (btags.value[t], get_norm_1d_csr(v, numOfUsers))) \
               .cache()
p_uh_co_occur = p_uh.map(lambda (h, v): (h, bp_h.value[h] * v)) \
                    .cache()


traces, entropy = split_entropy(1.0, 0.001, 0.1, numOfTags, bp_h.value, p_uh, p_uh_co_occur)

