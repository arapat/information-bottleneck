import numpy as np

def sum_of_tuples(tuple_array, value_func):
  return sum(map(value_func, tuple_array))

def normalize(tuple_array, total):
  return map(lambda (a, b): (a, 1.0 * b / total), tuple_array)

def vectorize(vals, dictionary):
  vector = []
  idx = 0
  for verb, val in vals:
    while dictionary[idx] != verb:
      vector.append(0.0)
      idx = idx + 1
    vector.append(val)
    idx = idx + 1
  vector = vector + [0.0] * (len(dictionary) - idx)
  return np.array(vector)

