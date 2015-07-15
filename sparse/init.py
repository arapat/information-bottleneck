import numpy as np

from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

def norm(p):
  """
  in-place normalize
  """
  assert p.dtype == 'float64'
  assert len(p.shape) == 2
  normalize(p, norm='l1', axis=1, copy=False)
  return p


def get_norm_1d_csr(idx_val, size):
  data = np.array([val for idx, val in idx_val])
  indices = np.array([idx for idx, val in idx_val])
  indptr = np.array([0, len(idx_val)])
  return csr_matrix((1.0 * data / data.sum(), indices, indptr), shape=(1, size))
 
