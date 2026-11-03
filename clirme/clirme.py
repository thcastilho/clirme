# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Thiago César Castilho Almeida et al.
#
# This file is an implementation of ClIR.ME: Clustering and Iterative Ranking for
# Meaningful Embedding.
#
# ClIR.ME is free software; you can redistribute it and/or modify it under the terms
# of the GNU General Public License as published by the Free Software Foundation;
# either version 2 of the License, or (at your option) any later version.
#
# ClIR.ME is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE. See the GNU General Public License for more details.

import numpy as np
import umap
from tqdm import tqdm
from sklearn.cluster import SpectralClustering, KMeans, HDBSCAN, AgglomerativeClustering
from finch import FINCH
from interpretable_embeddings.measures.qpp import compute_reciprocal_density, compute_accumulated_jacmax
from interpretable_embeddings.measures.correlation import (
    compute_jacmax, compute_jacmax_to_lists,
    compute_rbo, compute_rbo_to_lists,
    compute_reciprocal_knn_distance, compute_reciprocal_knn_distance_to_lists,
)
from .measures import compute_jaccard
from .utils import load_feat, compute_ranked_lists

CORRELATION_MEASURES = ('jacmax', 'jaccard', 'rbo', 'reciprocal')
ESTIMATION_MEASURES = ('reciprocal_density', 'accjacmax')
SELECTION_STRATEGIES = ('qpp', 'medoid', 'centroid', 'random')
CLUSTERING_ALGORITHMS = ('kmeans', 'hdbscan', 'agglomerative', 'spectral', 'finch')


class ClIRME:
    '''
    ClIR.ME: Clustering and Iterative Ranking for Meaningful Embedding.

    ALMEIDA, T. C. C. ; VARGAS, F. ; VALEM, L. P. ; PEDRONETTE, D. C. G.
    Self-Explaining Unsupervised Prototype-Based Embeddings via Projection and Ranking.
    Knowledge-Based Systems, 2026.

    Each dimension of the embedding is the rank correlation between an element and the
    prototype (leader) of one cluster. With skip_umap=False, the projection coordinates
    are appended after those dimensions.

    Args:
        feat_path (str or np.ndarray): Path to a .npy feature matrix [n_samples, n_features],
            or the matrix itself.
        d (int): Number of UMAP components of the projection.
        top_K (int): Depth of the ranked lists used by the QPP and correlation measures.
        correlation_measure (str): 'jacmax' (default), 'jaccard', 'rbo' or 'reciprocal'.
        estimation_measure (str): 'reciprocal_density' (default) or 'accjacmax'.
        alpha (float): Decay factor of 'accjacmax'.
        t (int): Number of iterations.
        skip_umap (bool): Use the L2-normalized features instead of the UMAP projection, and
            do not append the projection to the embedding.
        selection_strategy (str): 'qpp' (default: the element of each cluster with the
            highest estimation_measure score), 'medoid', 'centroid' or 'random'.
        rbo_p (float): Persistence parameter of 'rbo'.
        random_state (int): Seed of UMAP and of the 'random' selection strategy.
        verbose (bool): Print progress.
    '''

    def __init__(self, feat_path=None, d=2, top_K=20, correlation_measure='jacmax',
                 estimation_measure='reciprocal_density', alpha=0.95, t=3,
                 skip_umap=False, selection_strategy='qpp', rbo_p=0.9,
                 random_state=42, verbose=True):
        if feat_path is None:
            raise ValueError("Feature matrix (feat_path) path not set!")
        if correlation_measure not in CORRELATION_MEASURES:
            raise ValueError(f"Invalid correlation measure: {correlation_measure}")
        if estimation_measure not in ESTIMATION_MEASURES:
            raise ValueError(f"Invalid estimation measure: {estimation_measure}")
        if selection_strategy not in SELECTION_STRATEGIES:
            raise ValueError(f"Invalid selection_strategy: {selection_strategy}")

        self.verbose = verbose
        self._log("\tInitializing ClIR.ME")
        self.feat_path = feat_path
        self.features = load_feat(self.feat_path, verbose=verbose)
        self.d = d
        self.skip_umap = skip_umap
        self.selection_strategy = selection_strategy
        self.random_state = random_state

        if skip_umap:
            norms = np.linalg.norm(self.features, axis=1, keepdims=True)
            self.projection = self.features / np.where(norms == 0, 1, norms)
        else:
            self.projection = self.reduce_dimensionality()

        self.embeddings = []
        self.embeddings.append(self.projection)
        self.dataset_size = self.projection.shape[0]
        self.rks = None
        self.rks_array = None  # self.rks as an array, used by the vectorized measures

        self.t = t

        self.top_K = top_K
        self.correlation_measure = correlation_measure
        self.rbo_p = rbo_p
        self.correlation_matrix = {}  # cache of pairwise correlations, keyed by (i, j)
        self.alpha = alpha
        self.estimation_measure = estimation_measure
        self.estimations = None
        self.leaders = None

        self._log(f"\tProjection shape: {self.projection.shape}")

    def _log(self, message):
        if self.verbose:
            print(message)

    def reduce_dimensionality(self):
        reducer = umap.UMAP(n_components=self.d, random_state=self.random_state)
        return reducer.fit_transform(self.features)

    def cluster_features(self, clustering_algorithm='kmeans', params=None):
        """
        Cluster the current embedding (self.embeddings[-1]).

        Args:
            clustering_algorithm (str): 'kmeans', 'hdbscan', 'agglomerative', 'spectral' or
                'finch'.
            params (dict): Keyword arguments of the clustering algorithm (for 'finch',
                'req_clust' is required).

        Returns:
            tuple: (cluster label of each element, unique labels).
        """
        params = params or {}
        X = self.embeddings[-1]

        if clustering_algorithm == 'kmeans':
            labels = KMeans(**params).fit_predict(X)
        elif clustering_algorithm == 'hdbscan':
            labels = HDBSCAN(**params).fit(X).labels_
        elif clustering_algorithm == 'agglomerative':
            labels = AgglomerativeClustering(**params).fit(X).labels_
        elif clustering_algorithm == 'spectral':
            labels = SpectralClustering(**params).fit(X).labels_
        elif clustering_algorithm == 'finch':
            if params.get('req_clust') is None:
                raise ValueError("FINCH requires params={'req_clust': <number of clusters>}")
            c, num_clust, labels = FINCH(X, **params)
        else:
            raise ValueError(f"Invalid clustering algorithm: {clustering_algorithm}")

        unique_labels = np.unique(labels)
        self._log(f"\t\t{clustering_algorithm}: {len(unique_labels)} clusters")
        return labels, unique_labels

    def compute_estimations(self):
        """
        Score every ranked list with the QPP measure ('reciprocal_density' or
        'accjacmax') and store the scores in self.estimations.
        """
        self._log("\t\tComputing estimations")
        args = [(self.rks, i, self.top_K, self.alpha) for i in range(len(self.rks))]

        if self.estimation_measure == 'accjacmax':
            estimate = compute_accumulated_jacmax
        elif self.estimation_measure == 'reciprocal_density':
            estimate = compute_reciprocal_density
        else:
            raise ValueError(f"Invalid estimation measure: {self.estimation_measure}")

        self.estimations = [estimate(arg) for arg in tqdm(args, disable=not self.verbose)]

    def get_correlation(self, i, j):
        """
        Correlation between the ranked lists of elements i and j, cached per iteration.
        Symmetric measures share one cache entry for (i, j) and (j, i).

        Returns:
            float: Correlation score.
        """
        if self.correlation_measure in ('jacmax', 'jaccard', 'rbo'):
            key = (min(i, j), max(i, j))   # symmetric measures
        elif self.correlation_measure == 'reciprocal':
            key = (i, j)
        else:
            raise ValueError(f"Invalid correlation measure: {self.correlation_measure}")

        if key not in self.correlation_matrix:
            if self.correlation_measure == 'jacmax':
                value = compute_jacmax(self.rks[i], self.rks[j], self.top_K)
            elif self.correlation_measure == 'jaccard':
                value = compute_jaccard(self.rks[i], self.rks[j], self.top_K)
            elif self.correlation_measure == 'rbo':
                value = compute_rbo(self.rks[i], self.rks[j], self.top_K, self.rbo_p)
            else:  # 'reciprocal'
                value = compute_reciprocal_knn_distance(self.rks[i], self.rks[j], self.top_K, self.rks_array)
            self.correlation_matrix[key] = value

        return self.correlation_matrix[key]

    def get_correlations_to(self, leader):
        """
        get_correlation(i, leader) for every element i, in one vectorized call when the
        measure allows it.

        Returns:
            np.ndarray: One correlation score per element.
        """
        ref_list = self.rks_array[leader, :self.top_K]
        lists = self.rks_array[:, :self.top_K]

        if self.correlation_measure == 'jacmax':
            return compute_jacmax_to_lists(ref_list, lists)
        elif self.correlation_measure == 'rbo':
            return compute_rbo_to_lists(ref_list, lists, p=self.rbo_p)
        elif self.correlation_measure == 'reciprocal':
            # The element's list comes first in the pair, as in get_correlation(i, leader).
            return compute_reciprocal_knn_distance_to_lists(ref_list, self.rks_array, ref_first=False)
        else:  # 'jaccard'
            return np.array([self.get_correlation(i, leader) for i in range(self.dataset_size)])

    def compute_leaders(self, unique_labels, cluster_labels, estimations):
        """
        Pick one leader per cluster according to self.selection_strategy:

            'qpp': the element with the highest QPP score (default).
            'medoid': the element with the highest mean correlation to the rest of the cluster.
            'centroid': the element closest (L2) to the mean of the cluster's embeddings.
            'random': a random element, seeded with random_state.

        Negative labels (HDBSCAN noise) are not a cluster and get no leader.
        """
        leaders = []
        for cluster in tqdm(unique_labels, disable=not self.verbose):
            if cluster < 0:
                continue
            cluster_indexes = np.where(cluster_labels == cluster)[0]

            if self.selection_strategy == 'qpp':
                best_index = cluster_indexes[np.argmax(estimations[cluster_indexes])]

            elif self.selection_strategy == 'medoid':
                if len(cluster_indexes) == 1:
                    best_index = cluster_indexes[0]
                else:
                    mean_corrs = np.array([
                        np.mean([self.get_correlation(i, j) for j in cluster_indexes if j != i])
                        for i in cluster_indexes
                    ])
                    best_index = cluster_indexes[np.argmax(mean_corrs)]

            elif self.selection_strategy == 'centroid':
                cluster_embs = self.embeddings[-1][cluster_indexes]
                centroid = cluster_embs.mean(axis=0)
                dists = np.linalg.norm(cluster_embs - centroid, axis=1)
                best_index = cluster_indexes[np.argmin(dists)]

            elif self.selection_strategy == 'random':
                rng = np.random.default_rng(self.random_state)
                best_index = int(rng.choice(cluster_indexes))

            else:
                raise ValueError(f"Invalid selection_strategy: {self.selection_strategy}")

            leaders.append(best_index)
        self.leaders = leaders

    def get_leaders(self):
        """
        Return the list of leaders (prototypes) of the last iteration.

        Raises:
            ValueError: If leaders have not been computed yet.
        """
        if self.leaders is None:
            raise ValueError(
                "Leaders have not been computed yet. "
                "Call fit_transform() first."
            )
        return self.leaders

    def generate_embeddings(self, leaders):
        """
        Build the embedding of this iteration: one column per leader with the correlation
        of every element to it, followed by the projection coordinates unless
        skip_umap=True.
        """
        emb = np.zeros((self.dataset_size, len(leaders)))
        for idx, ld in enumerate(tqdm(leaders, disable=not self.verbose)):
            emb[:, idx] = self.get_correlations_to(ld)

        if self.skip_umap:
            combined = emb
        else:
            combined = np.hstack((emb, self.projection))
        self._log(f"\t\tEmbeddings shape: {combined.shape}")  # (dataset_size, num_leaders + d)
        self.embeddings.append(combined)

    def fit_transform(self, clustering_algorithm='kmeans', params=None):
        """
        Run the t iterations and return every embedding produced.

        Returns:
            list[np.ndarray]: [projection, embedding of iteration 1, ..., embedding of iteration t].
        """
        for t in range(self.t):
            self._log(f"\tIteration: {t+1}")
            self.correlation_matrix = {}

            self._log("\t\tComputing ranked lists")
            # The measures read only the first top_K neighbours, so ask for no more.
            self.rks = compute_ranked_lists(self.embeddings[-1], top_k=self.top_K)
            self.rks_array = np.asarray(self.rks)

            self.compute_estimations()
            estimations = np.array(self.estimations)

            self._log("\t\tClustering")
            cluster_labels, unique_labels = self.cluster_features(clustering_algorithm=clustering_algorithm, params=params)

            self._log("\t\tSelecting leaders")
            self.compute_leaders(unique_labels=unique_labels, cluster_labels=cluster_labels, estimations=estimations)

            self._log("\t\tGenerating embeddings")
            self.generate_embeddings(leaders=np.array(self.leaders))

        return self.embeddings
