import numpy as np

from scipy.stats import entropy

def compute_entropy(p):
  sp = sorted(p)
  total = len(sp)
  sep = [i for i in range(total-1) if sp[i] != sp[i+1]]
  if not sep:
    return 1.0
  stats = [(sep[0] + 1.0)] \
      + [sep[k+1] - sep[k] for k in range(len(sep)-1)] \
      + [total - sep[-1] - 1]
  return entropy(np.array(stats) / total, base = 2.0)


# TODO: collect information on free energy
def split_entropy(init_beta, converge_dist, split_dist, numOfX, \
    p_x, p_yx, p_yx_co_occur, trials = 10):
  entropy = []
  traces = ['' for k in range(numOfX)]
  loop = 0
  while loop < trials:
    loop = loop + 1
    log.info("Loop %d" % loop)

    init_p_tx, assignments = hartigan_twoCentroids(p_x, p_yx, numOfX)
    beta, p_tx = search_beta(init_p_tx, init_beta, converge_dist, split_dist, p_x, p_yx, p_yx_co_occur)

    for k, c in zip(range(numOfX), np.argmax(p_tx, axis=1)):
      traces[k] = traces[k] + "(%d)" % c

    entropy.append(compute_entropy(traces))
  return traces, entropy


def classify(traces):
  count = len(traces)
  items = sorted(zip(traces, range(count)))
  result = []
  current, collection = items[0][0], [items[0][1]]
  for trace, n in items[1:]:
    if current != trace:
      result.append(collection)
      current, collection = trace, []
    collection.append(n)
  result.append(collection)
  return result

