# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Thiago César Castilho Almeida et al.
#
# Rank correlation measures that interpretable-embeddings does not provide.
# JaccardMax, RBO and Reciprocal kNN are imported from that package.

def compute_jaccard(Ti, Tj, top_K):
    """
    Plain Jaccard index between two ranked lists (Jaccard, 1901).

    JaccardMax takes the best Jaccard value over all prefix depths. This function
    computes it at a single depth instead:

        J = |Ti_K intersect Tj_K| / |Ti_K union Tj_K|

    The sets are built incrementally, as in JaccardMax, so the two measures differ
    only in that maximization. The loop stops at depth top_K - 1, so each set holds
    the first top_K - 1 entries of its list.

    Args:
        Ti (list[int]): First ranked list.
        Tj (list[int]): Second ranked list.
        top_K (int): Depth of the lists.

    Returns:
        float: Jaccard index of the two prefixes.
    """
    set_Ti = set()
    set_Tj = set()
    intersect = 0

    for d in range(1, top_K):
        new_ti = Ti[d - 1]
        new_tj = Tj[d - 1]
        intersect += (new_ti in set_Tj) + (new_tj in set_Ti) + (new_ti == new_tj)
        set_Ti.add(new_ti)
        set_Tj.add(new_tj)

    union = 2 * (top_K - 1) - intersect
    if union <= 0:
        return 0.0
    return intersect / union
