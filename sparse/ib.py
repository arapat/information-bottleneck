from time import time

import numpy as np

from numpy import log2
from numpy import sum
from numpy.random import uniform
from scipy.sparse import csr_matrix

def distance(p, q, valid = True, is_vector = False):
  return jsd(p, q, valid, is_vector)


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

    max_diff = np.max(distance(p_tx, new_p_tx))
    if max_diff <= converge_dist:
      break

    p_tx = np.array(new_p_tx)

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


# Distance measurement

def jsd(p, q, valid, is_vector):
  """
  p: a vector, or a matrix
  q: a vector, or a matrix
  valid: valid[k] = +inf if the data point is invalid, 1.0 otherwise
  return the jsd between p and every vector in q
  """
  if valid == False:
    return np.array([np.inf] * q.shape[0])

  m = (p + q) / 2.0
  t1 = np.nan_to_num(p * log2(p / m))
  t2 = np.nan_to_num(q * log2(q / m))
  # p, q are both vectors
  if is_vector:
    return 0.5 * (t1.sum() + t2.sum())
  # otherwise
  return 0.5 * (t1.sum(axis=1) + t2.sum(axis=1))


# Unused

def _csr_jsd(p, q):
  """
  p, q: csr_matrix, shape = (1, d)
  return the JS divergence between p and q.
  """
  _pq = np.intersect1d(p.nonzero()[1], q.nonzero()[1])
  _p = p[0, _pq].data
  _q = q[0, _pq].data
  m = (_p + _q) / 2.0
  plog2 = np.log2(_p / m)
  qlog2 = np.log2(_q / m)
  psum = p.sum() - _p.sum()
  qsum = q.sum() - _q.sum()
  result = 0.5 * (_p.dot(plog2) + _q.dot(qlog2) + psum + qsum)
  return result
  """
  return simple_jsd(p.toarray()[0], q.toarray()[0])
  """

