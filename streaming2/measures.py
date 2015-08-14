import numpy as np

from scipy.stats import entropy

def get_free_energy(p_x, p_tx, p_yx, p_t, p_yt, beta):
  def compute_p_xt():
    return norm(p_tx.T * p_x)

  def get_entropy(p_xt):
    return -np.sum(p_t * np.sum(p_xt * log2(p_xt), axis=1))

  def get_distortion():
    p_yt_sum = np.sum(p_yt, axis=1)
    js_div = p_yx.map(lambda (a, v): (a, distance(v, p_yt, p_yt_sum)))
    d = js_div.map(lambda (a, js): p_x[a] * np.sum(p_tx[a, :] * js))
    return d.sum()

  distortion = get_distortion()
  entropy = get_entropy(compute_p_xt())
  return distortion - entropy / beta, distortion, entropy


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


def repeat_run(alpha, beta, converge_dist, batchSize, numOfBatches, sampleBatches, \
    pairsIndex, trails = 10):
  testBatches = 3
  data = get_data(testBatches - 1, batchSize, pairsIndex, fromStart = True)
  idx, ratio_x, ratio_xy, all_x, all_xy, \
      c_x, bp_x, p_yx, p_yx_co_occur = gen_data(docs, 0, 0)
  nx = len(idx)

  assigns = ["0."] * nx
  entropies = [0.0]

  avg = p_yx.map(lambda (a, b): csr_matrix(bp_x.value[a] * b)) \
            .sum().toarray()[0]
  p_tx = np.ones((1,nx))
  p_t = np.ones(1)
  p_yt = np.array([avg])
  init_fe = get_free_energy(bp_x.value, p_tx, p_yx, p_t, p_yt, beta)
  free_energy = [init_fe]

  def join_assigns(new, old):
    return map(lambda (a, b): a + b, zip(old, new))

  def evaluate(p_t, p_yt_co_occur):
    p_yt = norm(p_yt_co_occur)
    p_tx = membership_probability(p_t, p_yt, beta, p_x, p_yx)
    fe = get_free_energy(bp_x.value, p_tx, p_yx, p_t, p_yt, beta)

    free_energy.append(fe)
    assigns = join_assigns(hardcluster(p_tx), assigns)
    entropy.append(compute_entropy(assigns))

  loop = 0
  while loop < trials:
    loop = loop + 1
    log.info("Loop %d" % loop)

    p_t, p_yt_co_occur = run_streaming(alpha, beta, converge_dist, \
        batchSize, numOfBatches, sampleBatches, pairsIndex)
    evaluate(p_t, p_yt_co_occur)

  return traces, entropy

