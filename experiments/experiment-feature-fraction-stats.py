"""Repeated-measures ANOVA + Tukey HSD on experiment-feature-fraction-sweep.py's
results (feature_fraction_sweep.csv), sliced two ways:

1. Within each instance-scoring method (attention / logit / clustering),
   across feature fraction -- does that method's test-set instance ROC AUC
   change as the feature representation gets less complete? Subject =
   model_version (0-4). Fraction=1.0 only has feature_version=0 (5
   replicates total), while the other fractions have 5 feature_versions x 5
   model_versions (25 replicates) -- to give AnovaRM the fully balanced
   panel it requires, each (fraction, model_version) cell is collapsed to
   its mean over feature_version first. feature_version is a nested random
   effect (which random feature subset was drawn) that this comparison
   isn't about, so averaging it out is the right move, not a shortcut: at
   fraction=1.0 the "average" is just that one replicate, unchanged.

2. Within each fraction, across scoring method -- given a feature-fraction
   level, is one method's instance ROC AUC reliably higher than another's?
   Subject = replicate (feature_version, model_version pair). This is
   naturally paired and needs no aggregation: attention, logit, and
   clustering are all scored on the SAME trained model against the SAME
   test instances within a replicate row.

3. Within each bag-level metric (ROC AUC / balanced accuracy / average
   precision), across feature fraction -- same design and same rationale as
   analysis 1, just applied to the bag_* columns instead of the instance
   attn/logit/cluster columns. This is the one to cite for "bag performance
   doesn't drop significantly at moderate feature degradation" -- it isn't
   implied by analyses 1/2 (which are instance-level only), so it needs its
   own test.

4. Within each logit-attention correlation metric (Pearson / Spearman),
   across feature fraction -- same design again, applied to pearson_r /
   spearman_r. Answers "does the logit-attention relationship itself
   strengthen or weaken as features degrade," separate from either
   quantity's individual classification performance (analyses 1-3).

All three are one-way RM-ANOVAs (not a single higher-order ANOVA with a
fraction x method/metric interaction term) so that the Tukey HSD post-hoc
stays a simple, directly-interpretable pairwise table within each slice, run
separately per method/metric (analyses 1 and 3) or per fraction (analysis
2) -- not corrected for running multiple such ANOVAs, since Tukey HSD
already controls the family-wise error rate within each one.

Saves the full Tukey HSD pairwise tables to OUTPUTS_DIR/camelyon16/:
feature_fraction_stats_within_method.csv (analysis 1, ROC AUC),
feature_fraction_stats_within_method_ap.csv (analysis 1b, same design
applied to average precision instead), feature_fraction_stats_within_fraction.csv
(analysis 2, ROC AUC), feature_fraction_stats_within_fraction_ap.csv
(analysis 2b, average precision), feature_fraction_stats_bag_performance.csv
(analysis 3), and feature_fraction_stats_correlation.csv (analysis 4) -- and
logs the RM-ANOVA and Tukey HSD output for all of them in full. Analyses
1b/2b exist because average precision, not ROC AUC, is the more appropriate
instance-level metric here given the instance-level class imbalance (see
NOTES-manuscript-changes-summary.md) -- the ROC-AUC versions (1/2) are kept
for reference/comparison, not because they're the preferred metric.
"""
import os
import logging
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
from datetime import datetime

import dotenv; dotenv.load_dotenv(override = True)

import pandas as pd

from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd

LOG_DIR = Path(os.getenv("GLOBAL_LOG_DIR")) / "camelyon16"
LOG_DIR.mkdir(exist_ok = True, parents = True)
OUTPUTS_DIR = Path(os.getenv("OUTPUTS_DIR")) / "camelyon16"

log = logging.getLogger(__name__)
now = datetime.now().strftime("%Y-%m-%d %Hh%Mm")
logging.basicConfig(
    level = logging.INFO,
    filename = LOG_DIR / f"{now} {Path(__file__).stem}.log",
    format = "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S"
)

METHOD_COLS = {"attention": "attn_roc_auc", "logit": "logit_roc_auc", "clustering": "cluster_roc_auc"}
AP_METHOD_COLS = {"attention": "attn_average_precision", "logit": "logit_average_precision", "clustering": "cluster_average_precision"}
BAG_METRIC_COLS = {
    "bag_roc_auc": "bag_roc_auc",
    "bag_balanced_accuracy": "bag_balanced_accuracy",
    "bag_average_precision": "bag_average_precision",
}
CORRELATION_COLS = {"pearson": "pearson_r", "spearman": "spearman_r"}

def tukey_to_df(tukey, group_col_name: str, group_value) -> pd.DataFrame:
    table = tukey.summary().data
    result = pd.DataFrame(table[1:], columns = table[0])
    result.insert(0, group_col_name, group_value)
    return result

def within_group_across_fraction(df: pd.DataFrame, group_cols: dict, group_col_name: str, metric_label: str) -> pd.DataFrame:
    """RM-ANOVA + Tukey HSD across fraction, run separately for each column
    in group_cols (subject=model_version, feature_version averaged out --
    same design/rationale as analysis 1's docstring above). Shared by
    analysis 1 (ROC AUC) and 1b (average precision), same method columns,
    different metric."""
    tukey_tables = []
    for group_name, col in group_cols.items():
        cell_means = (
            df.groupby(["fraction_str", "model_version"])[col]
            .mean()
            .reset_index()
        )
        anova = AnovaRM(cell_means, depvar = col, subject = "model_version", within = ["fraction_str"]).fit()
        log.info("\n=== %s %s across feature fraction (RM-ANOVA, subject=model_version, n=5) ===\n%s", group_name, metric_label, anova)

        tukey = pairwise_tukeyhsd(
            endog = cell_means[col].to_numpy(),
            groups = cell_means["fraction_str"].to_numpy(),
            alpha = 0.05,
        )
        log.info("--- %s %s: Tukey HSD across fraction ---\n%s", group_name, metric_label, tukey)
        tukey_tables.append(tukey_to_df(tukey, group_col_name, group_name))

    return pd.concat(tukey_tables, ignore_index = True)

def within_fraction_across_method(df: pd.DataFrame, method_cols: dict, metric_label: str) -> pd.DataFrame:
    """RM-ANOVA + Tukey HSD across scoring method, run separately within
    each fraction (subject=replicate -- naturally paired, no aggregation
    needed, see analysis 2's docstring above). Shared by analysis 2 (ROC
    AUC) and 2b (average precision), same fraction grouping, different
    metric."""
    tukey_tables = []
    for fraction in sorted(df["fraction"].unique()):
        sub = df[df["fraction"] == fraction].copy()
        sub["replicate"] = sub["feature_version"].astype(str) + "_" + sub["model_version"].astype(str)

        long = sub.melt(
            id_vars = "replicate", value_vars = list(method_cols.values()),
            var_name = "method", value_name = metric_label,
        )
        long["method"] = long["method"].map({v: k for k, v in method_cols.items()})

        anova = AnovaRM(long, depvar = metric_label, subject = "replicate", within = ["method"]).fit()
        log.info(
            "\n=== fraction=%s: %s across scoring method (RM-ANOVA, subject=replicate, n=%d) ===\n%s",
            fraction, metric_label, sub.shape[0], anova
        )

        tukey = pairwise_tukeyhsd(
            endog = long[metric_label].to_numpy(),
            groups = long["method"].to_numpy(),
            alpha = 0.05,
        )
        log.info("--- fraction=%s: %s Tukey HSD across method ---\n%s", fraction, metric_label, tukey)
        tukey_tables.append(tukey_to_df(tukey, "fraction", fraction))

    return pd.concat(tukey_tables, ignore_index = True)

def main():
    csv_path = OUTPUTS_DIR / "feature_fraction_sweep.csv"
    df = pd.read_csv(csv_path)
    df["fraction_str"] = df["fraction"].astype(str)

    log.info("Loaded %d rows from %s (fractions: %s)", len(df), csv_path, sorted(df["fraction"].unique()))

    ### === Analysis 1: within each method, across fraction (ROC AUC) === ###
    within_group_across_fraction(df, METHOD_COLS, "method", "ROC AUC").to_csv(
        OUTPUTS_DIR / "feature_fraction_stats_within_method.csv", index = False)

    ### === Analysis 1b: within each method, across fraction (average precision) === ###
    within_group_across_fraction(df, AP_METHOD_COLS, "method", "average precision").to_csv(
        OUTPUTS_DIR / "feature_fraction_stats_within_method_ap.csv", index = False)

    ### === Analysis 2: within each fraction, across method (ROC AUC) === ###
    within_fraction_across_method(df, METHOD_COLS, "roc_auc").to_csv(
        OUTPUTS_DIR / "feature_fraction_stats_within_fraction.csv", index = False)

    ### === Analysis 2b: within each fraction, across method (average precision) === ###
    within_fraction_across_method(df, AP_METHOD_COLS, "average_precision").to_csv(
        OUTPUTS_DIR / "feature_fraction_stats_within_fraction_ap.csv", index = False)

    ### === Analysis 3: within each bag metric, across fraction === ###
    bag_tukey = []
    for metric, col in BAG_METRIC_COLS.items():
        cell_means = (
            df.groupby(["fraction_str", "model_version"])[col]
            .mean()
            .reset_index()
        )
        anova = AnovaRM(cell_means, depvar = col, subject = "model_version", within = ["fraction_str"]).fit()
        log.info("\n=== %s across feature fraction (RM-ANOVA, subject=model_version, n=5) ===\n%s", metric, anova)

        tukey = pairwise_tukeyhsd(
            endog = cell_means[col].to_numpy(),
            groups = cell_means["fraction_str"].to_numpy(),
            alpha = 0.05,
        )
        log.info("--- %s: Tukey HSD across fraction ---\n%s", metric, tukey)
        bag_tukey.append(tukey_to_df(tukey, "bag_metric", metric))

    pd.concat(bag_tukey, ignore_index = True).to_csv(
        OUTPUTS_DIR / "feature_fraction_stats_bag_performance.csv", index = False)

    ### === Analysis 4: within each correlation metric, across fraction === ###
    corr_tukey = []
    for metric, col in CORRELATION_COLS.items():
        cell_means = (
            df.groupby(["fraction_str", "model_version"])[col]
            .mean()
            .reset_index()
        )
        anova = AnovaRM(cell_means, depvar = col, subject = "model_version", within = ["fraction_str"]).fit()
        log.info("\n=== %s (logit-attention correlation) across feature fraction (RM-ANOVA, subject=model_version, n=5) ===\n%s", metric, anova)

        tukey = pairwise_tukeyhsd(
            endog = cell_means[col].to_numpy(),
            groups = cell_means["fraction_str"].to_numpy(),
            alpha = 0.05,
        )
        log.info("--- %s: Tukey HSD across fraction ---\n%s", metric, tukey)
        corr_tukey.append(tukey_to_df(tukey, "correlation_metric", metric))

    pd.concat(corr_tukey, ignore_index = True).to_csv(
        OUTPUTS_DIR / "feature_fraction_stats_correlation.csv", index = False)

    log.info("Done. Results at %s", OUTPUTS_DIR)

if __name__ == "__main__":
    main()
