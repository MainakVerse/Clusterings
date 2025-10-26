# 🤖 Machine Learning Clustering Algorithms

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-%F0%9F%A6%84-orange)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Contributions](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

A curated list of **20 clustering algorithms** implemented in or accessible via **Scikit-learn** 🧠  
These algorithms are widely used for **unsupervised learning**, **pattern discovery**, and **data segmentation**.

---

## 📊 Clustering Algorithms Overview

| 🔢 Serial No. | 🧩 Algorithm Name | 📦 Scikit-learn Import Path |
|:--------------:|:-----------------|:-----------------------------|
| 1 | K-Means | `from sklearn.cluster import KMeans` |
| 2 | MiniBatch K-Means | `from sklearn.cluster import MiniBatchKMeans` |
| 3 | Agglomerative Clustering | `from sklearn.cluster import AgglomerativeClustering` |
| 4 | DBSCAN | `from sklearn.cluster import DBSCAN` |
| 5 | OPTICS | `from sklearn.cluster import OPTICS` |
| 6 | Mean Shift | `from sklearn.cluster import MeanShift` |
| 7 | Spectral Clustering | `from sklearn.cluster import SpectralClustering` |
| 8 | Birch | `from sklearn.cluster import Birch` |
| 9 | Affinity Propagation | `from sklearn.cluster import AffinityPropagation` |
| 10 | Gaussian Mixture Model (GMM) | `from sklearn.mixture import GaussianMixture` |
| 11 | Bayesian Gaussian Mixture | `from sklearn.mixture import BayesianGaussianMixture` |
| 12 | Feature Agglomeration | `from sklearn.cluster import FeatureAgglomeration` |
| 13 | Bisecting K-Means | `from sklearn.cluster import BisectingKMeans` |
| 14 | K-Medoids | `from sklearn_extra.cluster import KMedoids` *(scikit-learn-extra)* |
| 15 | Fuzzy C-Means | `from fcmeans import FCM` *(external library)* |
| 16 | Self-Organizing Maps (SOM) | `from minisom import MiniSom` *(external library)* |
| 17 | HDBSCAN | `from hdbscan import HDBSCAN` *(external library)* |
| 18 | Spectral Biclustering | `from sklearn.cluster import SpectralBiclustering` |
| 19 | Spectral Coclustering | `from sklearn.cluster import SpectralCoclustering` |
| 20 | Ward Hierarchical Clustering | `from sklearn.cluster import AgglomerativeClustering` *(with linkage='ward')* |

---

## 🚀 Usage Example

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Sample data
X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

# Initialize and fit model
model = KMeans(n_clusters=3, random_state=42)
model.fit(X)

print(model.labels_)
```
