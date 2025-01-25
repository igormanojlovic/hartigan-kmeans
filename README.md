# hkmeans: Hartigan's K-Means in Python and in C++

[![license](https://img.shields.io/badge/License-Apache_2.0-brightgreen.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)

## Scope

This project provides an efficient implementation of Hartigan’s method for k-means clustering ([Hartigan 1975](#references)). It builds on the work of [Slonim, Aharoni and Crammer (2013)](#references), which introduced a significant improvement to the algorithm computational complexity, and adds an additional optimization for inputs in sparse vector representation. The project is packaged as a python library with a cython-wrapped C++ extension for the partition optimization code. A pure python implementation is included as well.


## Installation

```pip install hartigan-kmeans```

Alternatively:

```pip install git+https://github.com/igormanojlovic/hartigan-kmeans```


## Usage
The main class in this library is `HKmeans`, which implements the clustering interface of [SciKit Learn][sklearn], providing methods such as `fit()`, `fit_transform()`, `fit_predict()`, etc. 

The sample code below clusters the 18.8K documents of the 20-News-Groups dataset into 20 clusters:

```python

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn import metrics
from hkmeans import HKMeans

# read the dataset
dataset = fetch_20newsgroups(subset='all', categories=None,
                             shuffle=True, random_state=256)

gold_labels = dataset.target
n_clusters = np.unique(gold_labels).shape[0]

# create count vectors using the 10K most frequent words
vectorizer = TfidfVectorizer(max_features=10000)
X = vectorizer.fit_transform(dataset.data)

# HKMeans initialization and clustering; parameters:
# perform 10 random initializations (n_init=10); the best one is returned.
# up to 15 optimization iterations in each initialization (max_iter=15)
# use all cores in the running machine for parallel execution (n_jobs=-1)
hkmeans = HKMeans(n_clusters=n_clusters, random_state=128, n_init=10,
                  n_jobs=-1, max_iter=15, verbose=True)
hkmeans.fit(X)

# report standard clustering metrics
print("Homogeneity: %0.3f" % metrics.homogeneity_score(gold_labels, hkmeans.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(gold_labels, hkmeans.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(gold_labels, hkmeans.labels_))
print("Adjusted Rand-Index: %.3f" % metrics.adjusted_rand_score(gold_labels, hkmeans.labels_))
print("Adjusted Mutual-Info: %.3f" % metrics.adjusted_mutual_info_score(gold_labels, hkmeans.labels_))

```

Expected result:
```
Homogeneity: 0.245
Completeness: 0.290
V-measure: 0.266
Adjusted Rand-Index: 0.099
Adjusted Mutual-Info: 0.263
```

See the [Examples](examples) directory for more illustrations and a comparison against Lloyd's K-Means.


## Authors 
- Algorithm: [Hartigan 1975](#references)
- Pseudo-code and optimization: [Slonim, Aharoni and Crammer (2013)](#references)
- Programming, optimization and maintenance: [Assaf Toledo](https://github.com/assaftibm)


If you have any questions or issues you can create a new [issue here][issues].

## References
- Hartigan, John A. Clustering algorithms. Wiley series in probability and mathematical statistics: Applied probability and statistics. John Wiley & Sons, Inc., 1975.
- Slonim, Noam, Ehud Aharoni, and Koby Crammer. "Hartigan's K-Means Versus Lloyd's K-Means—Is It Time for a Change?." Twenty-Third International Joint Conference on Artificial Intelligence. 2013.


[issues]: https://github.com/IBM/hartigan-kmeans/issues/new
[sklearn]: https://scikit-learn.org
