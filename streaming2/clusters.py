
import numpy as np

def membership_probability(p_t, p_yt, beta, p_x, p_yx):
  p_yt_sum = np.sum(p_yt, axis=1)
  js_div = p_yx.map(lambda (a, v): (a, distance(v, p_yt, p_yt_sum)))
  result = js_div.map(lambda (a, v): (a, get_membership(v, p_t, beta))) \
      .sortByKey() \
      .map(lambda p: p[1]) \
      .collect()
  return result


def hardcluster(p, threshold = 0.9):
  """
  The last centroid will take the data point if no other centroid takes it.
  """
  psize = p.shape[1]
  def assign(dp):
    k = np.argmax(dp[:-1])
    if dp[k] >= threshold:
      return k
    return psize
  return map(assign, p)

