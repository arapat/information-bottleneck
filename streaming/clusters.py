
import numpy as np

def assignments(p_t, p_yt_co_occur, beta, min_diff, \
    p_x, p_yx):
  def assign(u):
    if u >= min_diff:
      return 1
    elif u <= -min_diff:
      return -1
    return 0

  p_yt = norm(p_yt_co_occur)
  p_yt_sum = np.sum(p_yt, axis=1)
  js_div = p_yx.map(lambda (a, v): (a, distance(v, p_yt, p_yt_sum)))
  result = js_div.map(lambda (a, v): (a, get_membership(v, p_t, beta))) \
      .sortByKey() \
      .map(lambda p: p[1]) \
      .map(lambda p: assign(p[0] - p[1])).collect()
  return result

