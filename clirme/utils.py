# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Thiago César Castilho Almeida et al.

import numpy as np
from sklearn.neighbors import BallTree

def load_feat(features, verbose=True):
    """
    Load the feature matrix, or return it unchanged if it is already an array.

    Args:
        features (str or np.ndarray): Path to a .npy file, or the matrix itself.
        verbose (bool): Print the path being read.

    Returns:
        np.ndarray: Feature matrix of shape [n_samples, n_features].
    """
    if isinstance(features, np.ndarray):
        return features
    if verbose:
        print("Reading features:", features)
    return np.load(features)

def compute_ranked_lists(features, top_k=1000):
    """
    Return the ranked list (indices of the nearest neighbours) of every element.

    The search uses FAISS ``IndexFlatL2`` if FAISS is installed and scikit-learn's
    ``BallTree`` if not. Both are exact L2 searches; FAISS runs in float32 and
    BallTree in float64, so they can differ by floating-point rounding. The main
    results of the paper were obtained with BallTree.

    The QPP and correlation measures read only the first ``top_K`` entries of each
    list, so pass that depth as ``top_k`` instead of keeping the default.

    Args:
        features (array-like of shape [n_samples, n_features]): Vectors to index.
        top_k (int): Number of neighbours per element, the element itself included.

    Returns:
        List[List[int]]: For each element, the indices of its top_k nearest
        neighbours, closest first.
    """
    n = features.shape[0]
    k = min(top_k, n)

    try:
        import faiss  # optional, installed with clirme[faiss]
        feats = np.ascontiguousarray(features, dtype=np.float32)
        index = faiss.IndexFlatL2(feats.shape[1])
        index.add(feats)
        _, ranked_lists = index.search(feats, k)
    except ImportError:
        tree = BallTree(features)
        _, ranked_lists = tree.query(features, k=k)

    return [list(map(int, neighbors)) for neighbors in ranked_lists]
