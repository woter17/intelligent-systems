"""Мой простой классификатор использует только длину лепестка."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


class PetalLengthRule(ClassifierMixin, BaseEstimator):
    """Сравниваю длину лепестка со средней длиной каждого вида на train.

    Ближайшее среднее определяет ответ. Для трёх упорядоченных средних
    это равносильно двум порогам между ними. Пороги получаются из train,
    а не берутся из теста или готового ответа.
    """
    def fit(self, X, y):
        values = np.asarray(X, dtype=float)
        labels = np.asarray(y)
        self.classes_ = np.unique(labels)
        self.n_features_in_ = values.shape[1]
        self.means_ = np.array([values[labels == label, 2].mean() for label in self.classes_])
        return self

    def predict(self, X):
        check_is_fitted(self, 'means_')
        # Третий столбец — petal_length. Остальные признаки это правило не использует.
        lengths = np.asarray(X, dtype=float)[:, 2]
        distances = abs(lengths[:, None] - self.means_[None, :])
        return self.classes_[distances.argmin(axis=1)]
