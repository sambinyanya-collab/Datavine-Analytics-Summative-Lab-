# Summative Lab — DataVine Analytics

Prototype machine learning solutions for three DataVine Analytics client projects, built as a junior
data scientist exercise covering classification, recommendation, and clustering — all using PCA for
dimensionality reduction.

## Contents

| File | Description |
|---|---|
| `DataVine_Analytics_Summative_Lab.ipynb` | Main notebook — all four lab steps |
| `wine.csv` | Wine dataset (chemical properties + class label) |
| `chickwts.csv` | Chick weight dataset (weight + feed type) |
| `USArrests.csv` | US state-level crime statistics |
| `requirements.txt` | Python dependencies |

## Client Projects

1. **Wine Classification System** — k-Nearest Neighbors classifier, tuned with `GridSearchCV`
   over k-value and distance metric, on PCA-reduced (95% variance retained) chemical features.
2. **Agricultural Feed Recommendation Engine** — PCA-reduced, standardized chick weights compared
   across feed types with cosine similarity to recommend comparable feeds.
3. **Regional Crime Pattern Analysis** — K-Means and Gaussian Mixture Model clustering on
   PCA-reduced (2 components) crime statistics, with the optimal number of clusters chosen via the
   elbow method (K-Means) and BIC (GMM).

## Workflow

Each project follows the same standardized pipeline: load and clean data → standardize numerical
features → apply PCA → tune/fit the model → evaluate and visualize results.

## Setup

```bash
pip install -r requirements.txt
```

Then open `DataVine_Analytics_Summative_Lab.ipynb` in Jupyter (the CSV files must sit in the same
directory as the notebook, since they're loaded with relative paths).

## Results Summary

| Project | Technique | Key Result |
|---|---|---|
| Wine Classification | PCA + k-NN | High test-set accuracy classifying wine cultivars |
| Feed Recommendation | PCA + cosine similarity | Feed types ranked by weight-gain performance similarity |
| Crime Pattern Analysis | PCA + K-Means / GMM | States grouped by crime profile, with GMM adding soft cluster-membership confidence |
