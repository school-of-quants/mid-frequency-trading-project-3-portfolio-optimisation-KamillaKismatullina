import datetime as dt
import json
from typing import Dict

from itertools import combinations
import pandas as pd
import numpy as np
from yaml import safe_load


def load_config(config_path: str) -> Dict:
    """Загружает yaml конфиг в виде python словаря

    Args:
        config_path (str): Путь до конфига

    Returns:
        Dict: Словарь с параметрами конфига
    """
    with open(config_path) as file:
        config = safe_load(file)
    return config


def save_dict(dict_, path):
    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(dict_, f, indent=4, default=str)


def applyPtSlOnT1(close, events, ptSl):
    """Tripple barrier method adjusted for Low and High prices

    Args:
        close (pd.Series): Close prices.
        events (pd.DataFrame): A pandas dataframe, with columns:
            - t1: The timestamp of vertical barrier. When the value is np.nan, there will not be a vertical barrier.
            - trgt: The unit width of the horizontal barriers.
        ptSl (list):
            - ptSl[0]: The factor that multiplies trgt to set the width of the upper barrier.
                If 0, there will not be an upper barrier.
            - ptSl[1]: The factor that multiplies trgt to set the width of the lower barrier.
                If 0, there will not be a lower barrier.

    Returns:
        pd.DataFrame: Timestamps of each barrier touch
    """

    # apply stop loss/profit taking, if it takes place before t1 (end of event)

    out = events[["t1"]].copy(deep=True)
    if ptSl[0] > 0:
        pt = ptSl[0] * events["trgt"]
    else:
        pt = pd.Series(index=events.index)  # NaNs
    if ptSl[1] > 0:
        sl = -ptSl[1] * events["trgt"]
    else:
        sl = pd.Series(index=events.index)  # NaNs
    for loc, t1 in events["t1"].fillna(close.index[-1]).items():
        df0 = close[loc:t1]  # path prices
        df0 = df0 / close[loc] - 1  # path returns
        out.loc[loc, "sl"] = df0[df0 < sl[loc]].index.min()  # earliest stop loss.
        out.loc[loc, "pt"] = df0[df0 > pt[loc]].index.min()  # earliest profit taking.
    return out


def three_barrier(close, ptSl=[1, 1], rolling_n=50, scaling_factor=2.0):
    """Labeling based on tripple barrier method and historical volatility

    Args:
        close (pd.DataFrame): Assets close prices
        ptSl (list, optional): applyPtSlOnT1 ptSl parameter. Defaults to [1, 1].
        rolling_n (int, optional): Rolling window to calculate standart deviation. Defaults to 50.
        scaling_factor (float, optional): Multiplier of standart deviation to calculate horizontal barrier size. Defaults to 2.0.

    Returns:
        pd.Series: 3-class labels
    """
    events = pd.DataFrame(
        {
            "t1": close.index + dt.timedelta(days=10),
            "trgt": 0.05,
        },
        index=close.index,
    )
    out = applyPtSlOnT1(close, events, ptSl)
    target = out.apply(
        lambda x: 1 if x.idxmin() == "pt" else -1 if x.idxmin() == "sl" else 0, axis=1
    )

    return target


class PurgedKFold:
    """
    K-Fold cross-validator with purging and embargo for time series.
    Args:
    n_splits : int
    purge_days : int
    embargo_days : int
    """
    
    def __init__(self, n_splits: int = 5, purge_days: int = 5, embargo_days: int = 2):
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days
    
    def split(self, X, y=None, groups=None):
        """
        Generate idx for train и validation.
        Returns:
        generator of (train_indices, test_indices)
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        fold_size = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            val_start = i * fold_size
            val_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples
            test_idx = indices[val_start:val_end]
            
            train_end = val_start - self.purge_days
            train_idx = indices[:max(0, train_end)]
            
            if self.embargo_days > 0 and i > 0:
                embargo_end = val_start - self.embargo_days
                train_idx = train_idx[train_idx < embargo_end]
            
            if len(train_idx) == 0:
                train_idx = np.array([], dtype=int)
            
            yield train_idx, test_idx
    
    def get_n_splits(self):
        return self.n_splits


def combinatorial_purged_cv(X, y, n_splits: int = 5, purge_days: int = 5, embargo_days: int = 2):
    """
    Combinatorial Purged Cross-Validation.   
    Returns:
    list of (train_indices, test_indices) tuples
    """
    
    n_samples = len(X)
    indices = np.arange(n_samples)
    fold_size = n_samples // n_splits
    
    folds = []
    for i in range(n_splits):
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_splits - 1 else n_samples
        folds.append(indices[start:end])
    
    cv_pairs = []
    for test_combo in combinations(range(n_splits), n_splits // 2):
        test_idx = np.concatenate([folds[i] for i in test_combo])
        train_idx = np.concatenate([folds[i] for i in range(n_splits) if i not in test_combo])
        
        min_test_date = min(test_idx)
        train_idx = train_idx[train_idx < (min_test_date - purge_days)]
        
        if len(train_idx) > 0:
            cv_pairs.append((train_idx, test_idx))
    
    return cv_pairs