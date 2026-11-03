# ClIR.ME

[![License](https://img.shields.io/badge/license-GPL--2.0--or--later-blue.svg)](https://github.com/thcastilho/clirme/blob/main/LICENSE)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Official_Repo-blue?logo=github)](https://github.com/thcastilho/clirme)

ClIR.ME (*Clustering and Iterative Ranking for Meaningful Embedding*) is the official implementation of the method from:

> Almeida, T. C. C., Vargas, F., Valem, L. P., Pedronette, D. C. G.
> *Self-Explaining Unsupervised Prototype-Based Embeddings via Projection and Ranking.*
> Knowledge-Based Systems, 2026.

ClIR.ME turns pre-trained deep features into sparse embeddings that you can read, and it needs no labels. Each dimension is the rank correlation between an element and a prototype, which is an actual element of the collection, so you can open it and look at it.

## Overview

The features are projected once with UMAP. Then, for each of the `t` iterations, ClIR.ME:

1. computes the top-K nearest neighbours of every element on the current embedding;
2. scores each ranked list with an unsupervised query performance prediction (QPP) measure, Reciprocal Density or Accumulated JaccardMax;
3. clusters the current embedding with KMeans, HDBSCAN, Agglomerative, Spectral or FINCH;
4. picks one leader per cluster, by default the element with the highest QPP score;
5. represents each element by its rank correlation to every leader (JaccardMax by default), followed by its projection coordinates.

JaccardMax, RBO, Reciprocal kNN, Reciprocal Density and Accumulated JaccardMax come from the [`interpretable-embeddings`](https://github.com/thcastilho/interpretable-embeddings) package.

## Installation

```bash
pip install clirme
```

Or the development version from GitHub:

```bash
pip install git+https://github.com/thcastilho/clirme.git
```

ClIR.ME needs Python 3.10 or newer and depends on `interpretable-embeddings>=2.0.0`, `numpy`, `tqdm`, `scikit-learn`, `umap-learn` and `finch-clust`.

Nearest neighbours are found with scikit-learn's `BallTree`, or with FAISS's `IndexFlatL2` if [FAISS](https://github.com/facebookresearch/faiss) is installed. Both are exact L2 searches. FAISS is faster on large collections but works in float32, so the two can differ by floating-point rounding. The main results of the paper were obtained with `BallTree`; if you want to reproduce them, don't install FAISS. Otherwise:

```bash
pip install "clirme[faiss]"
```

Expect small numerical differences from the values in the paper, mostly caused by dependency versions. UMAP projections in particular change between versions of `umap-learn`, `pynndescent` and `numba`.

## Usage

The input is a feature matrix of shape `[n_samples, n_features]`, for example features from a pre-trained ViT or ResNet. You can pass a path to a `.npy` file or the NumPy array itself.

```python
from clirme import ClIRME

clirme = ClIRME(
    feat_path="features.npy",          # or a NumPy array
    d=2,                               # UMAP components
    top_K=20,                          # depth of the ranked lists
    correlation_measure="jacmax",      # "jacmax", "jaccard", "rbo" or "reciprocal"
    estimation_measure="reciprocal_density",  # or "accjacmax"
    t=3,                               # iterations
)

embeddings = clirme.fit_transform(
    clustering_algorithm="kmeans",
    params={"n_clusters": 50, "random_state": 42},
)

# embeddings = [projection, embedding at t=1, ..., embedding at t=3]
X = embeddings[-1]                     # shape: (n_samples, n_clusters + d)

# Leaders of the last iteration: column i of X is the correlation to element prototypes[i]
prototypes = clirme.get_leaders()
```

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `feat_path` | required | Path to a `.npy` feature matrix, or the matrix itself. |
| `d` | `2` | Number of UMAP components. |
| `top_K` | `20` | Depth of the ranked lists used by the QPP and correlation measures. The paper uses 100 for Corel5k, 200 for Pets and 60 for CUB-200. |
| `correlation_measure` | `"jacmax"` | Rank correlation to the prototypes: `"jacmax"`, `"jaccard"`, `"rbo"` or `"reciprocal"`. |
| `estimation_measure` | `"reciprocal_density"` | QPP measure: `"reciprocal_density"` or `"accjacmax"`. |
| `alpha` | `0.95` | Decay factor of `"accjacmax"`. |
| `t` | `3` | Number of iterations. |
| `skip_umap` | `False` | Use the L2-normalized features instead of the UMAP projection. The projection is then not appended to the embedding. |
| `selection_strategy` | `"qpp"` | How each cluster's prototype is chosen: `"qpp"` takes the element with the highest `estimation_measure` score; the others are `"medoid"`, `"centroid"` and `"random"`. |
| `rbo_p` | `0.9` | Persistence parameter of `"rbo"`. |
| `random_state` | `42` | Seed for UMAP and for the `"random"` strategy. |
| `verbose` | `True` | Print progress. |

`fit_transform(clustering_algorithm, params)` takes `"kmeans"`, `"hdbscan"`, `"agglomerative"`, `"spectral"` or `"finch"`, and passes `params` to the matching scikit-learn class. FINCH needs `{"req_clust": n}`. The number of clusters is the number of prototypes, and so the number of interpretable dimensions. HDBSCAN noise points don't get a prototype of their own, but they are still embedded by their correlation to the other clusters' prototypes.

ClIR.ME is transductive. UMAP, the ranked lists and the prototypes are computed over the whole collection you pass to `fit_transform`, and there is no `transform` for unseen elements. To embed new elements, add them to the collection and run `fit_transform` again.

In retrieval, most of the effectiveness comes from the projection coordinates; the prototype dimensions are what make the embedding interpretable. Leave `skip_umap=False` unless you want prototype dimensions only, and expect that representation to be less effective.

## Examples

Retrieval:

```python
from sklearn.metrics.pairwise import cosine_similarity

query_idx = 0
sims = cosine_similarity(X)
top_k = sims[query_idx].argsort()[::-1][1:11]
print("Top-10 results for query:", top_k)
```

Classification:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, labels)
clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
print("Accuracy:", clf.score(X_test, y_test))
```

Explaining an element by its most similar prototypes:

```python
import numpy as np

i = 0
strongest = np.argsort(X[i, :len(prototypes)])[::-1][:5]
print("Most similar prototypes:", [prototypes[j] for j in strongest])
```

## Package structure

```
clirme/
├── clirme.py       # ClIR.ME implementation
├── measures.py     # Plain Jaccard index (other measures come from interpretable-embeddings)
└── utils.py        # Feature loading and ranked list computation (FAISS / BallTree)
```

## Citation

If you use this package in your research, please cite:

```bibtex
@article{almeida2026clirme,
  title   = {Self-Explaining Unsupervised Prototype-Based Embeddings via Projection and Ranking},
  author  = {Almeida, Thiago C{\'e}sar Castilho and Vargas, Francielle and Valem, Lucas Pascotti and Pedronette, Daniel Carlos Guimar{\~a}es},
  journal = {Knowledge-Based Systems},
  year    = {2026}
}
```

## Contact

- Thiago César Castilho Almeida: `tc.almeida@unesp.br`
- Lucas Pascotti Valem: `lucaspascottivalem@gmail.com`
- Daniel Carlos Guimarães Pedronette: `pedronette@gmail.com`
