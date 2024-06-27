from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np



def lda_split(X_s1 : np.array,
              X_s2 : np.array,
              y : np.array) -> np.array:


  best_n_components_s1 = get_best_n_components(X_s1, y)
  best_n_components_s2 = get_best_n_components(X_s2, y)


  best_n_components = min(best_n_components_s1, best_n_components_s2)

  X_lda_s1 = lda(X_s1, y , best_n_components)
  X_lda_s2 = lda(X_s2, y,  best_n_components)

  X = np.hstack((X_lda_s1, X_lda_s2))
  return X


def get_best_n_components(X : np.array,
                          y : np.array) -> int:
  lda = LinearDiscriminantAnalysis()

  X_lda = lda.fit_transform(X, y)

  explained_variance_ratio = lda.explained_variance_ratio_
  cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
  n_components = np.argmax(cumulative_variance_ratio >= 0.95) + 1

  return  n_components




def lda(X : np.array,
        y : np.array,
        n_components_best : int) -> np.array:

  lda = LinearDiscriminantAnalysis(n_components=n_components_best,
            svd_solver='auto')



  X_lda = lda.fit_transform(X, y)
  return X_lda