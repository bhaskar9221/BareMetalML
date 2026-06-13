import numpy as np
import sys
sys.path.append('/home/chotu/Projects/BareMetalML')


class GaussianNB:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing
        self.classes_ = None
        self.mean_ = None
        self.var_ = None
        self.priors_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.mean_ = {}
        self.var_ = {}
        self.priors_ = {}

        for c in self.classes_:
            X_c = X[y == c]
            self.mean_[c] = np.mean(X_c, axis=0)
            self.var_[c] = np.var(X_c, axis=0) + self.var_smoothing
            self.priors_[c] = len(X_c) / len(X)

    def _gaussian_log_pdf(self, x, mean, var):
        return -0.5 * np.log(2 * np.pi * var) - ((x - mean) ** 2) / (2 * var)

    def _predict_one(self, x):
        posteriors = []

        for c in self.classes_:
            prior = np.log(self.priors_[c])
            likelihood = np.sum(
                self._gaussian_log_pdf(x, self.mean_[c], self.var_[c])
            )
            posteriors.append(prior + likelihood)

        return self.classes_[np.argmax(posteriors)]

    def predict(self, X):
        return np.array([self._predict_one(x) for x in X])

class MultinomialNB:

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes_ = None
        self.feature_log_prob_ = None
        self.class_log_prior_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.feature_log_prob_ = {}
        self.class_log_prior_ = {}

        n_words = X.shape[1]

        for c in self.classes_:
            X_c = X[y == c]

            total_count_per_word = np.sum(X_c, axis=0)
            total_words_in_class = np.sum(total_count_per_word)

            word_probs = (
                total_count_per_word + self.alpha
            ) / (
                total_words_in_class + self.alpha * n_words
            )

            self.feature_log_prob_[c] = np.log(word_probs)
            self.class_log_prior_[c] = np.log(len(X_c) / len(X))

    def _predict_one(self, x):
        posteriors = []

        for c in self.classes_:
            log_posterior = (
                self.class_log_prior_[c]
                + np.sum(x * self.feature_log_prob_[c])
            )
            posteriors.append(log_posterior)

        return self.classes_[np.argmax(posteriors)]

    def predict(self, X):
        return np.array([self._predict_one(x) for x in X])