"""Statistics of the feature-fraction sweep for the JMI-260453 revision (reviewer 2, major 6).

1. Across fractions, for each scoring method and metric: a linear mixed model
   `value ~ C(fraction) + (1 | feature subset)`, fit by maximum likelihood. The
   models of one fraction share 5 random feature subsets (1 at fraction 1.0), and
   no model or subset is in two fractions, so this is not a repeated measure.
   Test: likelihood ratio against `value ~ 1`; pairwise fraction contrasts are
   Wald tests with a Holm correction within the method and metric.
2. Across methods, in each fraction: the methods score the same model on the same
   test instances, so the model is a true repeated measure. One-way RM-ANOVA, then
   paired t-tests of cluster_za against each other method (Holm in the fraction).

Reads sweep_ablation_metrics.csv (utils-sweep-ablation-metrics.py) and
feature_fraction_sweep.csv; writes feature_fraction_stats_r1.csv to OUTPUTS_DIR/camelyon16/.
Run from experiments/: python utils-sweep-ablation-stats.py
"""
import os, warnings
from itertools import combinations
from pathlib import Path

import dotenv; dotenv.load_dotenv(override = True)
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests

OUT = Path(os.getenv("OUTPUTS_DIR")) / "camelyon16"
THRESHOLD_METHODS = ["cluster_za", "cluster_z", "logit"]  # methods with a label-free decision


def load() -> pd.DataFrame:
    """One row per (checkpoint, method), with the bag metrics as the method 'bag'."""
    abl = pd.read_csv(OUT / "sweep_ablation_metrics.csv")
    bag = (
        pd.read_csv(OUT / "feature_fraction_sweep.csv")
        .rename(columns = {"bag_roc_auc": "roc_auc", "bag_average_precision": "average_precision",
                           "bag_balanced_accuracy": "balanced_accuracy"})
        .assign(method = "bag")
    )
    df = pd.concat([abl, bag[abl.columns.intersection(bag.columns)]], ignore_index = True)
    df["subset"] = df["fraction"].astype(str) + "_" + df["feature_version"].astype(str)
    df["model"] = df["subset"] + "_" + df["model_version"].astype(str)
    return df


def across_fraction(df: pd.DataFrame, method: str, metric: str) -> list[dict]:
    d = df[df["method"] == method].dropna(subset = [metric])
    with warnings.catch_warnings():
        # A subset variance of 0 (boundary fit) is expected; it is flagged in the `note` column.
        warnings.simplefilter("ignore")
        full = smf.mixedlm(f"{metric} ~ C(fraction)", d, groups = d["subset"]).fit(reml = False)
        null = smf.mixedlm(f"{metric} ~ 1", d, groups = d["subset"]).fit(reml = False)
    lr = 2 * (full.llf - null.llf)
    k = d["fraction"].nunique() - 1
    rows = [dict(analysis = "across_fraction", method = method, metric = metric, group = "all",
                 test = "mixedlm_lr", comparison = "fraction", stat = lr, df = k,
                 p = stats.chi2.sf(lr, k), effect = full.cov_re.iloc[0, 0],  # effect: subset variance
                 note = ("" if full.converged and null.converged else "not converged; ")
                        + ("boundary fit" if full.cov_re.iloc[0, 0] < 1e-8 else ""))]
    # Sensitivity: one-way ANOVA on the subset means (fraction 1.0 has one subset, so no df).
    means = d.groupby(["fraction", "subset"])[metric].mean()
    f = stats.f_oneway(*[g.to_numpy() for _, g in means.groupby(level = "fraction")])
    rows.append(dict(analysis = "across_fraction", method = method, metric = metric, group = "all",
                     test = "anova_subset_means", comparison = "fraction", stat = f.statistic,
                     df = f"{k}, {len(means) - k - 1}", p = f.pvalue, effect = np.nan))
    names = full.fe_params.index
    fracs = sorted(d["fraction"].unique())
    pair = []
    for f1, f2 in combinations(fracs, 2):
        c = np.zeros(len(names))
        for f, s in ((f2, 1), (f1, -1)):
            if f != fracs[0]:
                c[list(names).index(f"C(fraction)[T.{f}]")] += s
        t = full.t_test(c[None, :])
        pair.append(dict(analysis = "across_fraction", method = method, metric = metric, group = "all",
                         test = "wald", comparison = f"{f2} - {f1}", stat = float(t.tvalue.item()),
                         df = np.nan, p = float(t.pvalue), effect = float(t.effect.item())))
    holm = multipletests([r["p"] for r in pair], method = "holm")[1]
    return rows + [r | {"p_holm": h} for r, h in zip(pair, holm)]


def across_method(df: pd.DataFrame, fraction: float, metric: str, methods: list[str]) -> list[dict]:
    d = df[(df["fraction"] == fraction) & df["method"].isin(methods)]
    a = AnovaRM(d, depvar = metric, subject = "model", within = ["method"]).fit().anova_table.iloc[0]
    eta = a["F Value"] * a["Num DF"] / (a["F Value"] * a["Num DF"] + a["Den DF"])
    rows = [dict(analysis = "across_method", method = "+".join(methods), metric = metric, group = fraction,
                 test = "rm_anova", comparison = "method", stat = a["F Value"],
                 df = f"{a['Num DF']:.0f}, {a['Den DF']:.0f}", p = a["Pr > F"], effect = eta)]  # partial eta^2
    wide = d.pivot(index = "model", columns = "method", values = metric)
    pair = []
    for m in methods[1:]:
        diff = wide["cluster_za"] - wide[m]
        t = stats.ttest_rel(wide["cluster_za"], wide[m])
        pair.append(dict(analysis = "across_method", method = m, metric = metric, group = fraction,
                         test = "paired_t", comparison = f"cluster_za - {m}", stat = t.statistic,
                         df = len(diff) - 1, p = t.pvalue, effect = diff.mean()))
        if d["subset"].nunique() > 1:  # sensitivity: the 25 models are in 5 subsets
            sub = wide.groupby(wide.index.str.rsplit("_", n = 1).str[0]).mean()
            t = stats.ttest_rel(sub["cluster_za"], sub[m])
            pair.append(dict(analysis = "across_method", method = m, metric = metric, group = fraction,
                             test = "paired_t_subset_means", comparison = f"cluster_za - {m}", stat = t.statistic,
                             df = len(sub) - 1, p = t.pvalue, effect = (sub["cluster_za"] - sub[m]).mean()))
    holm = multipletests([r["p"] for r in pair], method = "holm")[1]
    return rows + [r | {"p_holm": h} for r, h in zip(pair, holm)]


def main():
    df = load()
    rank_methods = ["cluster_za"] + sorted(set(df["method"]) - {"cluster_za", "bag"})
    rows = []
    for metric in ["average_precision", "roc_auc"]:
        for m in rank_methods + ["bag"]:
            rows += across_fraction(df, m, metric)
        for f in sorted(df["fraction"].unique()):
            rows += across_method(df, f, metric, rank_methods)
    for metric in ["f1", "balanced_accuracy", "negative_bag_positive_rate"]:
        for m in THRESHOLD_METHODS:
            rows += across_fraction(df, m, metric)
        for f in sorted(df["fraction"].unique()):
            rows += across_method(df, f, metric, THRESHOLD_METHODS)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "feature_fraction_stats_r1.csv", index = False)
    with pd.option_context("display.width", 200, "display.max_rows", 500, "display.float_format", "{:.4g}".format):
        print(out[out["test"].isin(["mixedlm_lr", "rm_anova"])].drop(columns = "p_holm").to_string(index = False))


if __name__ == "__main__":
    main()
