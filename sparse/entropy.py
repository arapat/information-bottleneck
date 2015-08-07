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


def hard_clustering(p_tx):
  return np.argmax(p_tx, axis=1)


def split_entropy(p_tx, beta, converge_dist, split_dist, alpha, \
    p_x, p_yx, p_yx_co_occur, trials = 10):
  n = p_tx.shape[0]

  free_energy = []
  num_of_trials = []
  entropy = []
  traces = ['' for k in range(n)]
  loop = 0
  while loop < trials:
    loop = loop + 1
    log.info("Loop %d" % loop)

    succeed, result = fixed_beta_split(p_tx, beta, converge_dist, split_dist, alpha, \
        p_x, p_yx, p_yx_co_occur)
    new_p_tx, fe, trial_count = result
    for k, c in zip(range(n), hard_clustering(new_p_tx)):
      traces[k] = traces[k] + "(%d)" % c

    entropy.append(compute_entropy(traces))
    free_energy.append(fe)
    num_of_trials.append(trial_count)
  return traces, entropy, free_energy, num_of_trials


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

