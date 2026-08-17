"""Classes/methods/funnctions to preprocess/clean data.

Includes feature filtration/selection, as well as general data cleaning utiity.

"""

### === Dependencies === ###
import time, logging, functools # Short-hand for Native Python deps
from typing import Callable, Literal, Any

import numpy as np
import pandas as pd
import polars as pl
from polars import selectors as cs

from sklearn.impute import KNNImputer
from sklearn.feature_selection import mutual_info_classif

log = logging.getLogger(__name__)

# === Basic Data Cleaning Class === #
class DataCleaner:
    """Class for cleaning Dataframes in sklearn style with fit/transform syntax.
    """
    def __init__(self):
        """Instantiate and initialize 'trainable hyperparameters'
        """
        self.mode = "train"
        self.imputer = None
        self.col_means = None
        self.col_max = None
        self.col_min = None
        self.to_log: list[str] = None
        self.log_replaces_dist: bool = None
        self.train_first_error = (
            "%s is None - must run in training mode first."
        )

        return

    def set_mode(self, mode: Literal['train', 'test']) -> None:
        """Method to change the mode attribute, if needed.
        """
        self.mode = mode if mode in ['train', 'test'] else 'invalid'
        if self.mode == 'invalid':
            raise ValueError(f"{mode=} is invalid - must be 'train' or 'test'.")

        return

    def handle_outliers(
            self,
            df_x: pl.DataFrame,
            percentile_lower_bound: float,
            percentile_upper_bound: float,
            multiplier: float = 0.,
            outlier_handle_method: Literal['clip', 'remove'] = 'clip'
    ) -> pl.DataFrame:
        """Remove or clip elements from a distribution via percentile bounds.

        For IQR-based removal, set multiplier > 0. For pure percentile-based
        outlier removal, set multiplier = 0.

        Args:
            dist (np.ndarray): The distribution to remove outliers from.
            percentile_lower_bound (float): A percentile of the distribution to
                use as the lower bound.
            percentile_upper_bound (float): A percentile of the distribution to
                use as the upper bound.
            multiplier (float, optional): A multiplicative factor to the
                difference of percentile_lower_bound and percentile_upper_bound
                to determine the range of outliers to remove. Defaults to 1.5.
            method (Literal['clip', 'remove'], optional): Clip will set all
                outliers to the upper or lower bound. Remove will remove all
                outliers. Defaults to 'clip'.
                
        Returns:
            np.ndarray: The distribution with elements outside of the determined
                range removed.
        """
        if self.mode == "train":
            low_bounds = df_x.quantile(percentile_lower_bound / 100)
            high_bounds = df_x.quantile(percentile_upper_bound / 100)

            dist_range = high_bounds - low_bounds
            self.low_bounds = low_bounds - multiplier * dist_range
            self.high_bounds = high_bounds + multiplier * dist_range

        else:
            if self.low_bounds is None or self.high_bounds is None:
                raise ValueError(self.train_first_error % "bounds")

        if outlier_handle_method == 'clip':
            df_x = (
                df_x.with_columns(
                    pl.all().clip(self.low_bounds, self.high_bounds)
                )
            )

        elif outlier_handle_method == 'remove':
            df_x = (
                df_x.filter(
                    pl.all_horizontal(
                        cs.numeric().is_between(
                            cs.numeric().quantile(0.05),
                            cs.numeric().quantile(0.95)
                        )
                    )
                )
            )

        else:
            raise ValueError(f"{outlier_handle_method=} is invalid.")

        return df_x

    def inf_to_nearest_value(self, df_x: pl.DataFrame):
        """Convert inf or -inf values to nearest real value in distribution.

        Args:
            df_x (pl.DataFrame): Polars dataframe to process.

        Returns:
            pl.DataFrame: Processed Dataframe.
        """
        mx = df_x.to_numpy()

        # >> If in training mode, self attributes should be filled out for
        # >> future reference (i.e. `transform(...)`)
        if self.mode == 'train':
            finite_mx = np.where(np.isfinite(mx), mx, np.nan)
            self.col_max = np.nanmax(finite_mx, axis = 0)
            self.col_min = np.nanmin(finite_mx, axis = 0)
        elif (self.col_max is None) or (self.col_min is None):
            raise ValueError(self.train_first_error % "col_max/col_min")

        pos_inf_mask = np.isposinf(mx)
        mx[pos_inf_mask] = self.col_max[np.where(pos_inf_mask)[1]]

        neg_inf_mask = np.isneginf(mx)
        mx[neg_inf_mask] = self.col_min[np.where(neg_inf_mask)[1]]

        fixed_df = pl.DataFrame(mx, schema = df_x.columns)

        return fixed_df

    def fill_nan_with_feature_means(self, df_x: pl.DataFrame):
        """Mean imputation of features.
        """
        mx = df_x.to_numpy()
        if self.mode == "train":
            self.col_means = np.nanmean(mx, axis = 0)
        else:
            if self.col_means is None:
                raise ValueError(self.train_first_error % "col_mean")
        idx = np.where(np.isnan(mx))
        mx[idx] = self.col_means[idx[1]]

        fixed_df = pl.DataFrame(mx, schema = df_x.columns)

        return fixed_df

    def knn_imputation(
        self,
        df_x: pl.DataFrame,
        k: int = 10,
        weights: Literal['uniform', 'distance'] = "distance",
        **knn_imputer_kwargs
    ) -> pl.DataFrame:
        """Wrapper around KNNImputer.
        """
        mx_input = (
            df_x
            .with_columns(pl.all().cast(pl.Float64))
            .fill_null(float("nan"))   # Polars null → np.nan, not None
            .to_numpy()
        )

        if self.mode == "train":
            self.imputer = KNNImputer(
                n_neighbors = k, weights = weights, **knn_imputer_kwargs
            )
            mx = self.imputer.fit_transform(mx_input)
        else:
            if self.imputer is None:
                raise ValueError(self.train_first_error % "imputer")
            mx = self.imputer.transform(mx_input)

        fixed_df = pl.DataFrame(mx, schema = df_x.columns)

        return fixed_df

    def log_skewed_distributions(
        self,
        df_x: pl.DataFrame,
        skew_thresh: float = 1.0,
        improvement_thresh: float = 0.3,
        replace_feature: bool = True
    ) -> pd.DataFrame:
        """Identify and log features that would benefit from log transform.
        Relies on heuristic thresholds for skewness metrics.
        """
        if self.mode == "train":
            skew = df_x.skew(axis = 0)

            sig = skew[skew > skew_thresh]
            skew_after_log = np.log10(
                df_x[sig.index]
                + df_x[sig.index].min(axis = 0).abs()
                + 1e-8
            ).skew(axis = 0)

            improved = skew_after_log < (1 - improvement_thresh) * sig
            self.to_log = improved[improved].index.to_numpy()
            self.log_replaces_dist = replace_feature

            self.log_col_min = df_x[self.to_log].min(axis = 0)
            self.to_add_c = self.to_log[self.log_col_min < 0]
            self.min_c = 1.1 * self.log_col_min[self.to_add_c].abs() + 1e-8

        else:
            if self.log_replaces_dist is None or self.to_log is None:
                raise ValueError(self.train_first_error % "log attributes")

        new_names = list(map(lambda x: f"{x}_log", self.to_log))

        # >> Check negative values
        df_x[self.to_add_c] += self.min_c

        # >> Apply log transform
        df_x[new_names] = np.log10(df_x[self.to_log] + 1e-8)

        if self.log_replaces_dist:
            df_x = df_x.drop(self.to_log, axis = 1)

        return df_x

    def fit_transform(self, df_x: pl.DataFrame, **fit_kwargs) -> pl.DataFrame:
        """Fit DataCleaner instance and transform data.
        """
        # >> TODO: Add option to choose which imputations to apply.
        self.mode = "train"
        # Arg validation
        fit_kwargs['k'] = fit_kwargs.get('k', 10)
        fit_kwargs['weights'] = fit_kwargs.get('weights', 'distance')
        fit_kwargs['copy'] = fit_kwargs.get('copy', True)

        fit_kwargs['skew_thresh'] = fit_kwargs.get('skew_thresh', 1.0)
        fit_kwargs['improvement_thresh'] = fit_kwargs.get('improvement_thresh', 0.3)
        fit_kwargs['replace_feature'] = fit_kwargs.get('replace_feature', True)

        fit_kwargs['percentile_lower_bound'] = fit_kwargs.get(
            'percentile_lower_bound', 0.01 # 0.01 is left tail of z = 3
        )
        fit_kwargs['percentile_upper_bound'] = fit_kwargs.get(
            'percentile_upper_bound', 99.99 # 99.99 is right tail of z = 3
        )
        fit_kwargs['multiplier'] = fit_kwargs.get('multiplier', 0)
        fit_kwargs['outlier_handle_method'] = fit_kwargs.get(
            "outlier_handle_method", "clip"
        )

        knn_kwargs = {
            k : v for k,v in fit_kwargs.items()
            if k in ['k', 'weights', 'copy']
        }
        log_kwargs = {
            k : v for k,v in fit_kwargs.items()
            if k in ['skew_thresh', 'improvement_thresh', 'replace_feature']
        }
        outlier_kwargs = {
            k : v for k,v in fit_kwargs.items()
            if k in [
                'percentile_lower_bound', 'percentile_upper_bound',
                'multiplier', 'outlier_handle_method'
            ]
        }

        df_x = self.inf_to_nearest_value(df_x)
        df_x = self.knn_imputation(df_x, **knn_kwargs)
        # df_x = self.log_skewed_distributions(df_x, **log_kwargs)

        return df_x

    def fit(self, df_x: pl.DataFrame, **fit_kwargs) -> None:
        """In-place method to fit instance, wrapper of `fit_transform`.
        Returns None.
        """
        _ = self.fit_transform(df_x, **fit_kwargs)
        return

    def transform(self, df_x: pl.DataFrame) -> pl.DataFrame:
        self.mode = 'test'

        df_x = self.inf_to_nearest_value(df_x)
        df_x = self.knn_imputation(df_x)
        # df_x = self.log_skewed_distributions(df_x)

        return df_x

### === Feature Filtration Functions === ###

def get_mi_classification_scores(
        df_x: pl.DataFrame,
        y: pl.Series,
        random_state: int | None = None,
        **kwargs
) -> dict[str, float]:
    """
    Computes the mutual information classification score for each feature.
    """
    # Convert Polars objects to NumPy arrays for scikit-learn compatibility
    x_arr = df_x.to_numpy()
    y_arr = y.to_numpy()
    
    # Compute the mutual information scores
    mi_scores = (
        mutual_info_classif(
            X=x_arr, 
            y=y_arr, 
            random_state=random_state, 
            **kwargs
        )
    )
    
    # Zip the Polars column names with the resulting numpy array of scores
    return dict(zip(df_x.columns, mi_scores))


def get_correlated_feature_pairs(
    df_x: pl.DataFrame,
    corr_thresh: float = 0.95,
) -> list[tuple]:
    """
    Calculates the correlation matrix and returns pairs of features 
    that have an absolute correlation strictly greater than the threshold.
    """
    corr_df = df_x.corr()
    corr_df = corr_df.with_columns(pl.Series("feature_1", corr_df.columns))
    
    correlated_pairs = (
        corr_df
            .unpivot(
                index="feature_1", 
                variable_name="feature_2", 
                value_name="correlation"
            )
            .with_columns(pl.col("correlation").abs())
            .filter(
                (pl.col("correlation") > corr_thresh) & 
                (pl.col("feature_1") < pl.col("feature_2")) 
            )
    )
    
    return correlated_pairs.rows()

def filter_feature_pairs_by_metric(
        feature_pairs: list[tuple], 
        metric_vals: dict[str, float]
) -> list[str]:
    """
    Evaluates pairs of features and returns a list of features to drop,
    favoring the feature in the pair with the higher metric score.
    """
    # >> Sort for predictable behavior
    feature_pairs = sorted(
        feature_pairs,
        key = lambda x: (metric_vals[x[0]], metric_vals[x[1]])
    )

    to_drop = set()
    for pair in feature_pairs:
        v1, v2, _ = pair
        
        # If one is already marked for dropping, we can skip
        if v1 in to_drop or v2 in to_drop:
            continue

        # Drop the feature with the lower metric score
        if metric_vals[v1] >= metric_vals[v2]:
            to_drop.add(v2)
        else:
            to_drop.add(v1)
            
    return list(to_drop)

def filter_correlated_features_by_mi_score(
        df_x: pl.DataFrame,
        df_y: pl.Series,
        corr_thresh: float = 0.85,
        random_state: int | None = None
) -> pl.DataFrame:
    """
    Main pipeline function: Identifies highly correlated feature pairs, 
    evaluates them based on their Mutual Information scores, drops the 
    inferior features, and returns the cleaned DataFrame.
    """
    # 1. Get MI scores for all features
    mi_scores = (
        get_mi_classification_scores(
            df_x,
            df_y,
            random_state = random_state
        )
    )
    
    # 2. Get highly correlated pairs
    correlated_pairs = (
        get_correlated_feature_pairs(
            df_x,
            corr_thresh = corr_thresh
        )
    )
    
    # 3. Figure out which features to drop
    to_drop = filter_feature_pairs_by_metric(correlated_pairs, mi_scores)
    
    # 4. Drop the features (Note: no axis=1 needed in Polars)
    return df_x.drop(to_drop)
