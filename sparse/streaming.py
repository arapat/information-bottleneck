
def ib_streaming(p_tx, beta, converge_dist, split_dist, alpha, \
    p_x, p_yx, p_yx_co_occur, dict_x, trials = 10):
  nodes = [(p_tx, dict_x)]
  while True:
    log.info("ib_streaming: seaching beta")
    beta = search_beta(p_tx, 1.0, 0.001, 0.1, 0.01, bp_h.value, p_uh, p_uh_co_occur, \
      maximum_trial = 5)
    log.info("found beta %d" % beta)

    traces, entropies, free_energy, num_of_trials = split_entropy(p_tx, beta, 0.001, 0.1, 0.01, \
        bp_h.value, p_uh, p_uh_co_occur, trials = 20)


