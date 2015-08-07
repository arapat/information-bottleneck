
import numpy as np

def distance(p, q, valid = True, is_vector = False):
  return jsd(p, q, valid, is_vector)


# Distance measurement

def jsd(p, q, valid, is_vector):
  """
  p: a vector, or a matrix
  q: a vector, or a matrix
  valid: valid[k] = +inf if the data point is invalid, 1.0 otherwise
  return the jsd between p and every vector in q
  """
  assert(type(p) is np.ndarray)
  assert(type(q) is np.ndarray)
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

