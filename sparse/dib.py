from time import time

import numpy as np

from numpy import log2
from numpy import sum
from numpy.random import uniform
from scipy.sparse import csr_matrix


def get_free_energy(p_tx, p_t, p_x, js_div, beta):
  """
  t: centroids
  p_tx: P(T|X)
  p_x, p_tx, p_t, js_div: array
  """
  def compute_p_xt():
    raw = p_tx.T * p_x
    return norm(raw)

  def get_entropy(p_xt):
    """
    p_xt: array
    """
    return -np.sum(p_t * np.sum(p_xt * log2(p_xt), axis=1))

  def get_distortion():
    d = js_div.map(lambda (a, js): p_x[a] * np.sum(p_tx[a, :] * js))
    return d.sum()

  distortion = get_distortion()
  entropy = get_entropy(compute_p_xt())
  return distortion - entropy / beta, distortion, entropy, entropy / beta


def perturbate(p_tx, alpha):
  """
  perturb every leaf node to create two potential new centroids
  p_tx : array
  """
  n_x, c = p_tx.shape
  result = []
  for k in range(c):
    eps = alpha * uniform(-0.5, 0.5, n_x)
    result.append(p_tx.T[k] * (0.5 + eps))
    result.append(p_tx.T[k] * (0.5 - eps))
  result = np.array(result)
  return np.array(result.T)


def get_membership(js_div, p_t, beta):
  """
  return the probability of a specific data point x belongs to each centroid
  js_div: 1-d array of a specific data point x
  p_t: array
  """
  p = p_t * np.exp(-beta * js_div)
  if p.sum() == 0.0:
    return p
  return p / p.sum()


def converge(p_tx, beta, converge_dist, p_x, p_yx, p_yx_co_occur):
  """
  Run the iterative information until p_tx converges
  p_yx, p_yx_co_occur: csr_matrix
  p_tx, p_x: array
  """

  # To return
  p_yt = None
  iterations = 0
  # To compute free energy
  p_t = None
  js_div = None

  while True:
    iterations = iterations + 1
    
    p_t = p_tx.T.dot(p_x)

    p_yt_co_occur = p_yx_co_occur.map(lambda (a, v): np.outer(p_tx[a, :], v.toarray())) \
                                 .sum()
    p_yt = norm(p_yt_co_occur)

    # new p(t|x)
    if js_div:
      js_div.unpersist()
    js_div = p_yx.map(lambda (a, v): (a, distance(v.toarray(), p_yt, p_x[a] > 0.0))).cache()

    new_p_tx = js_div.map(lambda (a, v): (a, get_membership(v, p_t, beta))) \
                     .sortByKey() \
                     .map(lambda p: p[1]).collect()
    new_p_tx = np.array(new_p_tx)

    max_diff = np.max(distance(p_tx, new_p_tx))
    if max_diff <= converge_dist:
      break

    p_tx = new_p_tx

  free_energy = get_free_energy(p_tx, p_t, p_x, js_div, beta)
  js_div.unpersist()

  return p_yt, p_tx, free_energy, iterations


def fixed_beta_split(p_tx, beta, converge_dist, split_dist, alpha, p_x, p_yx, p_yx_co_occur, \
    maximum_trial = np.inf):
  trial_count = 0
  while trial_count < maximum_trial:
    trial_count = trial_count + 1
    log.info("trial %d, beta = %f" % (trial_count, beta))

    # perturbate and converge
    adjusted_p_tx = perturbate(p_tx, alpha)
    timer = time()
    p_yt, new_p_tx, free_energy, iterations = \
        converge(adjusted_p_tx, beta, converge_dist, p_x, p_yx, p_yx_co_occur)
    log.info("Converge time %f seconds (%d iterations)" % (time() - timer, iterations))

    js_distance = np.max(distance(p_yt[::2], p_yt[1::2]))
    if js_distance > split_dist:
      return (True, (new_p_tx, free_energy, trial_count))
  return (False, None)


def search_beta(p_tx, init_beta, converge_dist, split_dist, alpha, p_x, p_yx, p_yx_co_occur, \
    maximum_trial = 5):
  left = init_beta
  right = np.inf
  beta = init_beta
  while left + 1.0 < right:
    succeed, result = fixed_beta_split(p_tx, beta, converge_dist, split_dist, alpha, \
        p_x, p_yx, p_yx_co_occur, maximum_trial)
    if succeed:
      right = beta
    else:
      left = beta

    if not np.isinf(right):
      beta = (left + right) / 2.0
    else:
      beta = beta * 2.0
  return right

