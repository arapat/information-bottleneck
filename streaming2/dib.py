from time import time

import numpy as np

from numpy import log2
from numpy import sum
from numpy.random import uniform
from scipy.sparse import csr_matrix


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

  return p_yt, _p_yt_co_occur, _p_t, iterations


def search_beta(p_t, p_yt_co_occur, ratio_t, ratio_yt, init_beta, converge_dist, \
    p_x, p_yx, p_yx_co_occur):
  beta = init_beta
  _p_yt, _p_yt_co_occur, _p_t, iterations = \
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
  weights = p_yx.map(lambda (a, b): (a, distance(b.toarray()[0], bavg.value, is_vector=True))) \
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

