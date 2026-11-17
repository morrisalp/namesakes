"""Cross-validated evaluation of the probes (5-fold CV, as in the paper).

centroid_sim depends on the other names in the pool, so the inductive variants
recompute it per fold using only training-fold centroids (no test-set leakage).
"""
import numpy as np
from scipy import stats


def cv_ols_r2(X_cols: list, y: np.ndarray, k: int = 5,
              n_perm: int = 1000, random_state: int = 42) -> dict:
    """K-fold cross-validated OLS R² with 95% CI and permutation p-value.

    X_cols: list of 1-D arrays (one per predictor)
    y:      1-D target array
    k:      number of CV folds
    n_perm: permutations for p-value estimation (0 = skip)

    Returns dict: cv_r2_mean, cv_r2_std, cv_r2_ci95_low, cv_r2_ci95_high,
                  fold_r2s, perm_p.
    """
    from sklearn.model_selection import KFold
    from sklearn.linear_model import LinearRegression

    X = np.column_stack(X_cols) if len(X_cols) > 1 else np.asarray(X_cols[0]).reshape(-1, 1)
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    def _fold_r2s(X, y):
        r2s = []
        for train, test in kf.split(X):
            reg = LinearRegression().fit(X[train], y[train])
            ss_res = float(np.sum((y[test] - reg.predict(X[test])) ** 2))
            ss_tot = float(np.sum((y[test] - y[test].mean()) ** 2))
            r2s.append(1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0)
        return np.array(r2s)

    fold_r2s = _fold_r2s(X, y)
    mean_r2  = float(fold_r2s.mean())
    std_r2   = float(fold_r2s.std(ddof=1))
    ci_half  = float(stats.t.ppf(0.975, df=k - 1) * std_r2 / np.sqrt(k))

    perm_p = None
    if n_perm > 0:
        rng = np.random.default_rng(random_state)
        perm_means = np.array([
            _fold_r2s(X, rng.permutation(y)).mean()
            for _ in range(n_perm)
        ])
        perm_p = float(max(np.mean(perm_means >= mean_r2), 1.0 / n_perm))

    return {
        "cv_r2_mean":      mean_r2,
        "cv_r2_std":       std_r2,
        "cv_r2_ci95_low":  mean_r2 - ci_half,
        "cv_r2_ci95_high": mean_r2 + ci_half,
        "fold_r2s":        fold_r2s.tolist(),
        "perm_p":          perm_p,
    }


def _fold_censim(all_cn: np.ndarray, train_idx: np.ndarray,
                 pool_mask: np.ndarray = None) -> np.ndarray:
    """Recompute centroid_sim for all N names using only train-fold centroids as pool.

    For test names: max cos-sim to any training centroid.
    For training names: max cos-sim to any other training centroid (self excluded).
    pool_mask: optional bool array (N,); if given, only pool members used as reference.
    """
    train_cn = all_cn[train_idx]              # (n_train, d)
    sims = all_cn @ train_cn.T               # (N, n_train)
    for local_j, global_j in enumerate(train_idx):
        sims[global_j, local_j] = -np.inf    # exclude self
    if pool_mask is not None:
        for local_j, global_j in enumerate(train_idx):
            if not pool_mask[global_j]:
                sims[:, local_j] = -np.inf
    result = sims.max(axis=1)
    return np.where(result == -np.inf, 0.0, result)


def cv_ols_r2_inductive(fixed_cols: list, all_cn: np.ndarray, gt: np.ndarray,
                        k: int = 5, n_perm: int = 1000,
                        random_state: int = 42,
                        pool_mask: np.ndarray = None) -> dict:
    """cv_ols_r2 with fold-wise centroid_sim recomputation.

    fixed_cols: list of 1-D arrays for predictors that don't need recomputation (e.g. disp).
                Pass [] for censim-only regression.
    all_cn:     (N, d) L2-normalised centroid vectors, one per name.
    gt:         (N,) target array.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold

    n = len(gt)
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    fold_splits = list(kf.split(np.arange(n)))

    # Precompute inductive censim for each fold (reused across permutations)
    fold_data = []
    for train_idx, test_idx in fold_splits:
        censim = _fold_censim(all_cn, train_idx, pool_mask)
        cols_tr = [c[train_idx] for c in fixed_cols] + [censim[train_idx]]
        cols_te = [c[test_idx]  for c in fixed_cols] + [censim[test_idx]]
        X_tr = np.column_stack(cols_tr) if len(cols_tr) > 1 else cols_tr[0].reshape(-1, 1)
        X_te = np.column_stack(cols_te) if len(cols_te) > 1 else cols_te[0].reshape(-1, 1)
        fold_data.append((train_idx, test_idx, X_tr, X_te))

    def _run(y):
        r2s = []
        for train_idx, test_idx, X_tr, X_te in fold_data:
            reg = LinearRegression().fit(X_tr, y[train_idx])
            ss_res = float(np.sum((y[test_idx] - reg.predict(X_te)) ** 2))
            ss_tot = float(np.sum((y[test_idx] - y[test_idx].mean()) ** 2))
            r2s.append(1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0)
        return np.array(r2s)

    fold_r2s = _run(gt)
    mean_r2  = float(fold_r2s.mean())
    std_r2   = float(fold_r2s.std(ddof=1))
    ci_half  = float(stats.t.ppf(0.975, df=k - 1) * std_r2 / np.sqrt(k))

    perm_p = None
    if n_perm > 0:
        rng = np.random.default_rng(random_state)
        perm_means = np.array([_run(rng.permutation(gt)).mean() for _ in range(n_perm)])
        perm_p = float(max(np.mean(perm_means >= mean_r2), 1.0 / n_perm))

    return {
        "cv_r2_mean":      mean_r2,
        "cv_r2_std":       std_r2,
        "cv_r2_ci95_low":  mean_r2 - ci_half,
        "cv_r2_ci95_high": mean_r2 + ci_half,
        "fold_r2s":        fold_r2s.tolist(),
        "perm_p":          perm_p,
    }


def cv_separability_inductive(disp_r: np.ndarray, all_cn_r: np.ndarray,
                               disp_d: np.ndarray, all_cn_d: np.ndarray,
                               k: int = 5, random_state: int = 42,
                               pool_mask: np.ndarray = None) -> dict:
    """cv_separability with fold-wise centroid_sim recomputation.

    all_cn_r: (N, d) real-name centroids — used as the pool for all censim lookups.
    all_cn_d: (N, d) perturbed-name centroids — evaluated against the real training pool.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import KFold

    n = len(disp_r)
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    aucs, accs = [], []
    for fold_i, (train_idx, test_idx) in enumerate(kf.split(np.arange(n))):
        cen_r = _fold_censim(all_cn_r, train_idx, pool_mask)
        pool_train = train_idx[pool_mask[train_idx]] if pool_mask is not None else train_idx
        ref_cn = all_cn_r[pool_train] if len(pool_train) > 0 else all_cn_r[train_idx]
        cen_d = (all_cn_d @ ref_cn.T).max(axis=1)  # perturbed vs real pool; no self
        X_tr = np.vstack([
            np.column_stack([disp_r[train_idx], cen_r[train_idx]]),
            np.column_stack([disp_d[train_idx], cen_d[train_idx]]),
        ])
        y_tr = np.concatenate([np.ones(len(train_idx)), np.zeros(len(train_idx))])
        X_te = np.vstack([
            np.column_stack([disp_r[test_idx], cen_r[test_idx]]),
            np.column_stack([disp_d[test_idx], cen_d[test_idx]]),
        ])
        y_te = np.concatenate([np.ones(len(test_idx)), np.zeros(len(test_idx))])
        clf   = LogisticRegression(max_iter=1000, random_state=random_state).fit(X_tr, y_tr)
        proba = clf.predict_proba(X_te)[:, 1]
        aucs.append(float(roc_auc_score(y_te, proba)))
        accs.append(float((clf.predict(X_te) == y_te).mean()))

    result = {
        "cv_auc_mean":      float(np.mean(aucs)),
        "cv_auc_std":       float(np.std(aucs, ddof=1)),
        "cv_accuracy_mean": float(np.mean(accs)),
        "cv_accuracy_std":  float(np.std(accs, ddof=1)),
        "fold_aucs":        aucs,
        "fold_accuracies":  accs,
    }
    return result
