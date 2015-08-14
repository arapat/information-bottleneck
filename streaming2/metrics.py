
import numpy as np
from scipy.sparse import csr_matrix

def distance(p, q, qs = None, is_vector = False):
  if not qs is None:
    return csr_jsd(p, q, qs)
  return jsd(p, q, is_vector)


# Distance measurement

def jsd(p, q, is_vector):
  """
  p: a vector, or a matrix
  q: a vector, or a matrix
  return the jsd between p and every vector in q
  """
  assert(type(p) is np.ndarray)
  assert(type(q) is np.ndarray)

  m = (p + q) / 2.0
  t1 = np.nan_to_num(p * log2(p / m))
  t2 = np.nan_to_num(q * log2(q / m))
  # p, q are both vectors
  if is_vector:
    assert(len(p.shape) == 1)
    assert(p.shape == q.shape)
    return 0.5 * (t1.sum() + t2.sum())
  # otherwise
  return 0.5 * (t1.sum(axis=1) + t2.sum(axis=1))


def csr_jsd(p, q, qs):
  """
  p, q: csr_matrix, shape = (1, d)
  return the JS divergence between p and q.
  """
  assert(type(p) is csr_matrix)
  assert(type(q) is np.ndarray)
  assert(len(q.shape) == 2 and q.shape[0] > 1)
  _p = p.data
  _q = q[:, p.nonzero()[1]] # might be zero
  m = (_p + _q) / 2.0
  plog2 = np.log2(_p / m)
  qlog2 = np.log2(_q / m)
  psum = np.sum(_p * plog2, axis=1)
  qsum = qs - np.sum(_q, axis=1) + np.sum(np.nan_to_num(_q * qlog2), axis=1)
  return 0.5 * (psum + qsum)

