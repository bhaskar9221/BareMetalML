# BareMetalML

## Project Overview

`BareMetalML` is a repository for implementing machine learning models from scratch in Python. The aim is to learn the underlying algorithms and build a reusable codebase without depending on `scikit-learn` inside the model implementations.

## What this repo contains

- A planned implementation path for foundational utilities, models, and visual helpers.
- A focus on NumPy-based algorithms and math-first implementations.
- External validation and comparison against standard libraries only in tests or notebooks.

## Planned package layout

The project is expected to grow into the following structure:

- `baremetalml/`
  - `utils.py`
  - `linear_model.py`
  - `logistic.py`
  - `decision_tree.py`
  - `random_forest.py`
  - `knn.py`
  - `naive_bayes.py`
  - `kmeans.py`
  - `pca.py`
- `tests/` — validation scripts for each implementation
- `notebooks/` — interactive demos and the final comparison notebook

> Note: currently only the README exists in the repo. Implementation modules and tests will be added as the project progresses.

## Implementation roadmap

This repository follows a learning path from utilities to advanced models. Each major step is designed to build on the previous one.

### 1. Utils

Create the base tools used by all models:
- deterministic train/test splitting
- regression and classification metrics
- preprocessing helpers for normalization and standardization
- decision boundary plotting utilities

### 2. Linear Regression

Implement regression from scratch using both closed-form normal equations and gradient descent, then add Ridge regularization. The goal is to reproduce the behavior of `sklearn.linear_model.LinearRegression` and compare results.

### 3. Logistic Regression

Build binary and multiclass logistic regression with stable sigmoid activation, binary cross-entropy loss, and One-vs-Rest multiclass support.

### 4. Decision Tree

Implement a decision tree classifier from first principles using impurity metrics, best-split search, and recursive tree building.

### 5. Random Forest

Build an ensemble of decision trees with bootstrap sampling, feature subsampling, and majority-vote aggregation.

### 6. KNN

Implement k-nearest neighbors for classification and regression, including Euclidean and Manhattan distance metrics.

### 7. Naive Bayes

Implement Gaussian Naive Bayes with numerical stability, plus an optional multinomial variant for count-based classification.

### 8. K-Means

Implement K-Means clustering with random and k-means++ initialization, inertia tracking, and elbow analysis.

### 9. PCA

Implement Principal Component Analysis using covariance decomposition, dimensionality reduction, and explained variance analysis.

### 10. Final Comparison Notebook

Create a polished notebook that compares multiple models on the same datasets, reports performance metrics, and visualizes decision boundaries.

## Development notes

- The repository is currently a roadmap and planning document.
- The main implementation packages will be added under `baremetalml/`.
- Tests and notebooks will be added as the algorithms are implemented.
- `scikit-learn` will only be used for dataset loading, validation, and comparison.

## Next step

The  utility module (`baremetalml/utils`) is implemented along with the initial regression/classification metrics and Linear Regression Module. Now I will be implementing Logistic Regression.
