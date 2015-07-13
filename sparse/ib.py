from time import time

import numpy as np

from numpy import log2
from numpy import sum
from numpy.random import uniform
from scipy.sparse import csr_matrix

def simple_jsd(p, q):
  assert type(p) is np.ndarray
  assert type(q) is np.ndarray
  assert len(p.shape) == 1
  assert p.shape == q.shape

  m = (p + q) / 2.0
  t1 = np.nan_to_num(p * log2(p / m))
  t2 = np.nan_to_num(q * log2(q / m))
  return 0.5 * (t1.sum() + t2.sum())


def jsd(p, q):
  """
  p, q: csr_matrix, shape = (1, d)
  return the JS divergence between p and q.
  """
  assert type(p) is csr_matrix
  assert type(q) is csr_matrix
  assert p.shape[0] == 1
  assert p.shape == q.shape

  return simple_jsd(p.toarray()[0], q.toarray()[0])
  """
  result = 0.0
  data1, indices1, indptr1 = p.data, p.indices, p.indptr
  data2, indices2, indptr2 = q.data, q.indices, q.indptr
  i1, r1, i2, r2 = 0, 0, 0, 0
  ptr1, ptr2 = 1, 1

  while i1 < data1.size and i2 < data2.size:
    while i1 == indptr1[ptr1]:
      r1 = r1 + 1
      ptr1 = ptr1 + 1
    while i2 == indptr2[ptr2]:
      r2 = r2 + 1
      ptr2 = ptr2 + 1
    idx1 = (r1, indices1[i1])
    idx2 = (r2, indices2[i2])
    if idx1 < idx2:
      result = result + data1[i1]
      i1 = i1 + 1
    elif idx1 > idx2:
      result = result + data2[i2]
      i2 = i2 + 1
    else:
      m = (data1[i1] + data2[i2]) / 2.0
      result = result + data1[i1] * log2(data1[i1] / m) + data2[i2] * log2(data2[i2] / m)
      i1, i2 = i1 + 1, i2 + 1
  result = result + sum(data1[i1:]) + sum(data2[i2:])
  return result / 2.0
  """


def compute_jsd(v, centroids, valid):
  """
  v: array-like csr_matrix
  centroids: csr_matrix, a list of vectors
  return the jsd between v and every centroid
  """
  c = centroids.shape[0]
  if not valid:
    return np.array([np.inf] * c)
  return np.array([jsd(v, centroids[k, :]) for k in range(c)])


def get_centroids_weights(p_tx, p_x):
  assert type(p_tx) is np.ndarray
  assert type(p_x) is np.ndarray
  return p_tx.T.dot(p_x)


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
    
    p_t = get_centroids_weights(p_tx, p_x)

    p_yt_co_occur = p_yx_co_occur.map(lambda (a, v): np.outer(p_tx[a, :], v.toarray())) \
                                 .sum()
    p_yt = csr_matrix(norm(p_yt_co_occur))

    # new p(t|x)
    if js_div:
      js_div.unpersist()
    js_div = p_yx.map(lambda (a, v): (a, compute_jsd(v, p_yt, p_x[a] > 0.0))).cache()
    new_p_tx = js_div.map(lambda (a, v): (a, get_membership(v, p_t, beta))) \
                     .sortByKey() \
                     .map(lambda p: p[1]).collect()

    max_diff = 0.0
    for k in range(p_tx.shape[0]):
      if max_diff <= converge_dist:
        diff = simple_jsd(p_tx[k, :], new_p_tx[k])
        max_diff = max(diff, max_diff)
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

    js_distance = 0.0
    for k in range(0, p_yt.shape[0], 2):
      if js_distance <= split_dist:
        js_distance = jsd(p_yt[k, :], p_yt[k + 1, :])

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
