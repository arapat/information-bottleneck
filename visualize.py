from operator import itemgetter

def format_membership(p_cn_trans):
    """
    Return sorted elements for each cluster
    """
    result = []
    for c in range(len(p_cn_trans)):
        weight = [(nouns[n], p_cn_trans[c][n]) for n in range(len(nouns))]
        result.append(sorted(weight, key = itemgetter(1), reverse=True))
    return result

def plot_three_level(k, membership):
    """
    BUG: Cannot work, because p_cn for parent nodes are cleared.
    """
    total = len(membership)
    level = 0
    q = [(k, level, "")]
    while len(q) > 0:
        u, l, desc = q[0]
        del q[0]
        if l > level:
            print "============================="
            level = l
        if level + 1 < 3 and k * 2 + 1 < total:
            q.append((u * 2 + 1, level + 1, desc + "l"))
        if level + 1 < 3 and k * 2 + 2 < total:
            q.append((u * 2 + 2, level + 1, desc + "r"))
        
        print "(%d)" % u, "{", desc,
        print [p[0] for p in membership[u][:10]],
        print sum([1 for p in membership[u] if p[1] >= 0.8]),
        print "}"

