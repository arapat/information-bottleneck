from scipy.sparse import csr_matrix

def sum_of_tuples(tuple_array, value_func):
    return sum(map(value_func, tuple_array))

def normalize(tuple_array, total):
    return map(lambda (a, b): (a, 1.0 * b / total), tuple_array)

def vectorize(vals, dictionary):
    col_ind = []
    idx = 0
    for verb, val in vals:
        while dictionary[idx] != verb:
            idx = idx + 1
        col_ind.append(idx)
        idx = idx + 1
    row_ind = [0] * len(col_ind)
    return csr_matrix(([val for verb, val in vals], (row_ind, col_ind)), \
            shape=(1, len(dictionary)))

