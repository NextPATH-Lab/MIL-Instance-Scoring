"""LDO mixture components for the JMI-260453 revision (reviewer 2, major 7; Figure 3B).

Reads ldo_nucleus_components.csv (experiment-ldo-abmilite-training.py). The positive
component (LDO+) has the highest mean logit; LDO- is the other high-attention component,
with the lowest mean logit.
1. The fraction of LDO+ and LDO- nuclei of each patient, compared between the outcome
   groups (Mann-Whitney U and AUC), for train and test separately.
2. The panel B list: a seeded random sample of the confident members of LDO+ and of LDO-
   (posterior >= MIN_POSTERIOR; train and test patients), at most PER_PATIENT from each
   patient, N_EXAMPLES per cluster (spares included). A random sample, not the highest
   posterior (it saturates at 1 for outliers) or the nearest to the mean (LDO+ overlaps the
   bulk component there). (voron_x, voron_y) is the position in the biopsy image.
Run from experiments/: python utils-ldo-components.py
"""
import json, os
from pathlib import Path

import dotenv; dotenv.load_dotenv(override = True)
import numpy as np
import polars as pl
from scipy.stats import mannwhitneyu

OUT = Path(os.getenv("OUTPUTS_DIR")) / "LDO-Prostate"
N_EXAMPLES, PER_PATIENT, MIN_POSTERIOR, SEED = 24, 2, 0.9, 2380

means = np.array(json.load(open(OUT / "ldo-za-bgm-components.json"))["means_scaled_za"])
pos, neg = int(means[:, 0].argmax()), int(means[:, 0].argmin())
nuclei = pl.read_csv(OUT / "ldo_nucleus_components.csv")

per_patient = (
    nuclei.group_by("patient_id", "split", "DW2Y")
    .agg(n = pl.len(), ldo_pos = (pl.col("component") == pos).mean(), ldo_neg = (pl.col("component") == neg).mean())
    .sort("patient_id")
)
per_patient.write_csv(OUT / "ldo_patient_fractions.csv")

rows = []
for split in ["train", "test"]:
    for col in ["ldo_pos", "ldo_neg"]:
        d = per_patient.filter(pl.col("split") == split)
        agg, ind = d.filter(pl.col("DW2Y") == 1)[col].to_numpy(), d.filter(pl.col("DW2Y") == 0)[col].to_numpy()
        u = mannwhitneyu(agg, ind, alternative = "two-sided")
        rows.append(dict(split = split, fraction = col, n_aggressive = len(agg), n_indolent = len(ind),
                         median_aggressive = np.median(agg), median_indolent = np.median(ind),
                         U = u.statistic, p = u.pvalue, auc = u.statistic / (len(agg) * len(ind))))
tests = pl.DataFrame(rows)
tests.write_csv(OUT / "ldo_patient_fraction_tests.csv")

examples = pl.concat([
    nuclei.filter((pl.col("component") == k) & (pl.col(f"p{k}") >= MIN_POSTERIOR) & (pl.col("split") != "none"))
    .sample(fraction = 1, shuffle = True, seed = SEED)
    .group_by("patient_id", maintain_order = True).head(PER_PATIENT)
    .head(N_EXAMPLES)
    .select(pl.lit(name).alias("cluster"), "patient_id", "split", "DW2Y", "BiopsyID", "row",
            "voron_x", "voron_y", pl.col(f"p{k}").alias("posterior"), "z_scaled", "a_scaled")
    for k, name in [(pos, "LDO+"), (neg, "LDO-")]
])
examples.write_csv(OUT / "ldo_panel_b_examples.csv")

with pl.Config(tbl_rows = 40, tbl_cols = 12, tbl_width_chars = 200, float_precision = 3):
    print(f"LDO+ = component {pos}, LDO- = component {neg}\n", tests, "\n", examples)
