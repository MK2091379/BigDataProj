import numpy as np
import pandas as pd

def generate_outlier_dataset(n_samples=20000, n_features=2000, outlier_fraction=0.01, 
                             inlier_dist=('normal', {'loc':0, 'scale':1}), 
                             outlier_dist=('uniform', {'low':-10, 'high':10}),
                             output_path='outlier_dataset_20000_2000.csv'):
    """
    Generates an outlier detection dataset of shape (n_samples, n_features+1),
    where the last column 'label' is 0 for inliers and 1 for outliers.
    
    Parameters
    ----------
    n_samples : int
        Total number of samples (rows).
    n_features : int
        Number of features (columns, excluding the label).
    outlier_fraction : float
        Fraction of samples to mark as outliers.
    inlier_dist : tuple
        ('normal', params) or ('laplace', params) — distribution and its kwargs for inliers.
    outlier_dist : tuple
        ('uniform', params) or ('normal', params) — distribution and its kwargs for outliers.
    output_path : str
        File path for the resulting CSV (includes header row).
    """
    n_outliers = max(1, int(n_samples * outlier_fraction))
    n_inliers = n_samples - n_outliers
    
    # Generate inliers
    kind, params = inlier_dist
    if kind == 'normal':
        inliers = np.random.normal(size=(n_inliers, n_features), **params)
    elif kind == 'laplace':
        inliers = np.random.laplace(size=(n_inliers, n_features), **params)
    else:
        raise ValueError("Unsupported inlier distribution")
    
    # Generate outliers
    kind_o, params_o = outlier_dist
    if kind_o == 'uniform':
        outliers = np.random.uniform(size=(n_outliers, n_features), **params_o)
    elif kind_o == 'normal':
        outliers = np.random.normal(size=(n_outliers, n_features), **params_o)
    else:
        raise ValueError("Unsupported outlier distribution")
    
    # Combine and shuffle
    X = np.vstack([inliers, outliers])
    y = np.hstack([np.zeros(n_inliers, dtype=int), np.ones(n_outliers, dtype=int)])
    perm = np.random.permutation(n_samples)
    X, y = X[perm], y[perm]
    
    # Build DataFrame and save
    cols = [f"f{i}" for i in range(n_features)] + ['label']
    df = pd.DataFrame(np.column_stack([X, y]), columns=cols)
    #df.to_csv(output_path, index=False)
    print(f"Saved {n_samples}×{n_features} dataset ({n_outliers} outliers) to '{output_path}'")

if __name__ == "__main__":
    generate_outlier_dataset()
