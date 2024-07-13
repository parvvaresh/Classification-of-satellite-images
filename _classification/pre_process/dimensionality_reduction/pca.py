import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



def pca_split(X_s1 : np.array,
              X_s2 : np.array,
              ) -> np.array:


  best_n_components_s1 = get_best_n_components(X_s1)
  best_n_components_s2 = get_best_n_components(X_s2)


  best_n_components = min(best_n_components_s1, best_n_components_s2)

  X_pca_s1 = pca(X_s1, best_n_components)
  X_pca_s2 = pca(X_s2, best_n_components)

  X = np.hstack((X_pca_s1, X_pca_s1))
  return X


def get_best_n_components(X : np.array) -> int:

  pca = PCA().fit(X)
  cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
  best_n_components = np.argmax(cumulative_variance >= 0.95) + 1
  return best_n_components



def pca(X : np.array,
        n_components_best : int) -> np.array:

  if isinstance(n_components_best, type(None)):
    best_n_components = get_best_n_components(X)

  pca = PCA(n_components=n_components_best,
            svd_solver='auto')

  X_pca = pca.fit_transform(X)
  return X_pca