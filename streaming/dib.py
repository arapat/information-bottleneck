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


def converge(p_t, p_yt_co_occur, sum_x, sum_xy, \
    beta, converge_dist, \
    p_x, p_yx, p_yx_co_occur)
  """
  Run the iterative information until p_tx converges
  p_yx, p_yx_co_occur: csr_matrix
  p_tx, p_x: array
  """

  # To compute free energy
  js_div = None

  iterations = 0
  while True:
    iterations = iterations + 1
    
    p_yt = norm(p_yt_co_occur)
    # p(t|x)
    if js_div:
      js_div.unpersist()
    p_yt_sum = np.sum(p_yt, axis=1)
    js_div = p_yx.map(lambda (a, v): (a, distance(v, p_yt, p_yt_sum))).cache()

    p_tx = js_div.map(lambda (a, v): (a, get_membership(v, p_t, beta))) \
                     .sortByKey() \
                     .map(lambda p: p[1]).collect()
    p_tx = np.array(p_tx)

    p_t = p_tx.T.dot(p_x)

    p_yt_co_occur = p_yx_co_occur.map(lambda (a, v): np.outer(p_tx[a, :], v.toarray())) \
                                 .sum()

    #max_diff = np.max(distance(p_tx, new_p_tx))
    #if max_diff <= converge_dist:
    #  break

    #p_tx = new_p_tx

  free_energy = get_free_energy(p_tx, p_t, p_x, js_div, beta)
  js_div.unpersist()

  return p_yt, p_yt_co_occur, p_t, free_energy, iterations


def search_beta(p_t, p_yt_co_occur, sum_x, sum_xy, \
    init_beta, converge_dist, split_dist, \
    p_x, p_yx, p_yx_co_occur):
  left = init_beta
  right = np.inf
  beta = init_beta
  while left + 1.0 < right:
    p_yt, p_yt_co_occur, p_t, free_energy, iterations = \
        converge(p_t, p_yt_co_occur, sum_x, sum_xy, \
        beta, converge_dist, \
        p_x, p_yx, p_yx_co_occur)
    js_distance = np.max(distance(p_yt[::2], p_yt[1::2]))
    if js_distance > split_dist:
      right = beta
    else:
      left = beta

    if not np.isinf(right):
      beta = (left + right) / 2.0
    else:
      beta = beta * 2.0

  p_yt, p_yt_co_occur, p_t, free_energy, iterations = \
      converge(p_t, p_yt_co_occur, sum_x, sum_xy, \
      beta, converge_dist, \
      p_x, p_yx, p_yx_co_occur)
  return right, p_t, p_yt_co_occur

