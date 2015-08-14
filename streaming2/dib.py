from time import time

import numpy as np

from numpy import log2
from numpy import sum
from numpy.random import uniform
from scipy.sparse import csr_matrix


# def get_free_energy(p_tx, p_t, p_x, js_div, beta):
#   """
#   t: centroids
#   p_tx: P(T|X)
#   p_x, p_tx, p_t, js_div: array
#   """
#   def compute_p_xt():
#     raw = p_tx.T * p_x
#     return norm(raw)
# 
#   def get_entropy(p_xt):
#     """
#     p_xt: array
#     """
#     return -np.sum(p_t * np.sum(p_xt * log2(p_xt), axis=1))
# 
#   def get_distortion():
#     d = js_div.map(lambda (a, js): p_x[a] * np.sum(p_tx[a, :] * js))
#     return d.sum()
# 
#   distortion = get_distortion()
#   entropy = get_entropy(compute_p_xt())
#   return distortion - entropy / beta, distortion, entropy, entropy / beta


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


def converge(p_t, p_yt_co_occur, ratio_t, ratio_yt, beta, converge_dist, \
    p_x, p_yx, p_yx_co_occur):
  """
  Run the iterative information until p_tx converges
  p_yx, p_yx_co_occur: csr_matrix
  p_tx, p_x: array
  """

  # To compute free energy
  js_div = None

  _p_yt_co_occur = np.copy(p_yt_co_occur)
  _p_t = p_t
  p_yt = None
  iterations = 0
  while True:
    iterations = iterations + 1
    
    _p_yt = norm(_p_yt_co_occur)
    if not p_yt is None:
      js_distance = np.max(distance(p_yt, _p_yt))
      if js_distance <= converge_dist:
        break
      log.info("iteration %d: distance %f" % (iterations, js_distance))
    p_yt = _p_yt

    # p(t|x)
    p_yt_sum = np.sum(p_yt, axis=1)
    js_div = p_yx.map(lambda (a, v): (a, distance(v, p_yt, p_yt_sum))) #.cache()
    p_tx = js_div.map(lambda (a, v): (a, get_membership(v, _p_t, beta))) \
                     .sortByKey() \
                     .map(lambda p: p[1]).collect()
    p_tx = np.array(p_tx)

    _p_t = ratio_t * p_t + p_tx.T.dot(p_x)
    _p_t = _p_t / _p_t.sum()

    temp = p_yx_co_occur.map(lambda (a, v): \
        csr_matrix(np.outer(p_tx[a, :], v.toarray()))).sum()
    _p_yt_co_occur = ratio_yt * p_yt_co_occur + temp.toarray()

  #TODO: How to compute free energy?
  free_energy = 0.0 #get_free_energy(p_tx, p_t, p_x, js_div, beta)
  # js_div.unpersist()

  return p_yt, _p_yt_co_occur, _p_t, free_energy, iterations


def search_beta(p_t, p_yt_co_occur, ratio_t, ratio_yt, init_beta, converge_dist, \
    p_x, p_yx, p_yx_co_occur):
  beta = init_beta
  _p_yt, _p_yt_co_occur, _p_t, free_energy, iterations = \
      converge(p_t, p_yt_co_occur, ratio_t, ratio_yt, beta, converge_dist, \
      p_x, p_yx, p_yx_co_occur)
  _movement = np.max(distance(p_yt, _p_yt))
  _distance = np.max(distance(p_yt[::2], p_yt[1::2]))
  log.info("move: %f, dist: %f" % (_movement, _distance))
  return _p_t, _p_yt_co_occur, _movement, _distance

def init_centroids(c_x, p_x, p_yx, p_yx_co_occur, beta):
  # generate centroids
  avg = p_yx.map(lambda (a, b): csr_matrix(p_x[a] * b)) \
            .sum().toarray()[0]
  bavg = sc.broadcast(avg)
  weights = p_yx.map(lambda (a, b): (a, distance(b.toarray()[0], bavg.value))) \
                .map(lambda (a, d): (a, d * np.log2(c_x[a]))) \
                .sortByKey().map(lambda p: p[1]).collect()
  weights = np.array(weights)
  idx = np.random.choice(weights.size, p = weights / weights.sum())
  vec = p_yx.filter(lambda (a, b): a == idx).first()[1].toarray()[0]
  p_yt = np.array([vec, avg])

  # compute assignments
  p_t = np.array([0.5, 0.5])
  p_yt_sum = np.sum(p_yt, axis=1)
  js_div = p_yx.map(lambda (a, v): (a, distance(v, p_yt, p_yt_sum))) #.cache()
  p_tx = js_div.map(lambda (a, v): (a, get_membership(v, p_t, beta))) \
                   .sortByKey() \
                   .map(lambda p: p[1]).collect()
  p_tx = np.array(p_tx)

  # return
  p_yt_co_occur = p_yx_co_occur.map(lambda (a, v): \
      csr_matrix(np.outer(p_tx[a, :], v.toarray()))).sum().toarray()
  p_t = p_tx.T.dot(p_x)
  p_t = p_t / p_t.sum()

  return p_t, p_yt_co_occur, idx

