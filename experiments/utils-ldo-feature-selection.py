"""Script to select features from LDO data.

Features are selected by dropping features with a lower mutual information score
within a high-correlation feature pair, until no high correlation feature pairs
remain.

We use bag-labels as instance pseudo-labels in this case to compute mutual info
score. Feature selection is based on the training patients of low-risk vs GS9
high-risk patient classification.

NOTE:
There are some features we want to remove (as they are auxiliary features such as spatial location, or parameters used in feature calculation, or features with known instability) before any computational feature selection. These are listed below:
| Feature name | Description of Reason | 
| ---- | ---- | 
| P_* | Any feature starting with 'P_' indicates a parameter used for feature calculation |
| GROUP | Categorical variable indicating it's grouping. May include later | 
| locks | supplemental feature used in legacy software, not for predictive purposes | 
| random | same as above | 
| voron_x | centroid x coordinate | 
| voron_y | centroid y coordinate | 
| x_centroid | center of mass of mask. This is useless as we correct the coordinates so that it is centered | 
| y_centroid | same as above | 
| harmon[06-12]_fft | very noisy, as these features are very high-order FFTs | 
| diff_orient | measure of orientation of the mask | 
| Duration | Days until clinical endpoint for patient. This should be used not to predict, but as a target |
| Progression | Whether the patient experienced prostate cancer progression. Same as above | 
| BCR | Whether the patient experienced biochemical recurrence. Same as above | 
| DW2Y | Whether the patient died within 2 years of initial consultation. Same as above | 
| Poor Outcome | whether the patient experienced any poor clinical outcome. Same as above | 
"""

### === Dependencies === ###
import os, json, joblib, logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

import polars as pl
from polars import selectors as cs

import torch as th

from src.datasets import LDODataset
from src.preprocessing import filter_correlated_features_by_mi_score

### === Configure Script Environment === ###
load_dotenv(override = True)

LOG_DIR = Path(os.getenv("GLOBAL_LOG_DIR")) / "LDO-Prostate"
LOG_DIR.mkdir(exist_ok = True, parents = True)
CONFIG_DIR = Path(os.getenv("CONFIG_DIR"))
CONFIG_DIR.mkdir(exist_ok = True, parents = True)
OUTPUT_DIR = Path(os.getenv("OUTPUTS_DIR")) / "LDO-Prostate"
(OUTPUT_DIR / "figures").mkdir(exist_ok = True, parents = True)

PATIENT_ID_COL = os.getenv("PATIENT_ID_COL")
TRAINING_DATA_DIR = Path(os.getenv("TRAINING_DATA_DIR"))

### === Set up logger === ###
log = logging.getLogger(__name__)
now = datetime.now().strftime("%Y-%m-%d %Hh%Mm")
logging.basicConfig(
    level = logging.INFO,
    filename = LOG_DIR / f"{now} {Path(__file__).stem}.log", 
    format = "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S"
)

data_files = list(TRAINING_DATA_DIR.glob("*.csv"))
log.info("Found %d patient files.", len(data_files))

# === Retrieve Only Relevant Features  === ###
features_to_drop = [
    # Clinical features we want to keep later but don't want in selection
    "Duration", "Progression", "BCR", "DW2Y", "Poor Outcome",
    "Age", "GleasonScore", "MaxiGS", "TStageEnum", "iPSA", "damico_enum",
    # Auxiliary features for in-house software, not actual features
    "random", "locks", "BiopsyID", "Visit_No",
    # Unstable/meaningless features
    "diff_orient", "center_of_grav",
]
# Features which can be excluded using regex
exclusion_regex = (
    r"^(voron_.*|.*Micron|P_.*|.*_centroid|harmon(0[4-9]|1[0-2])_fft)$"
)

# >> Get the actual remaining features from an example file
example_data = pl.read_csv(data_files[0])
feature_names = (
    example_data
        .drop(features_to_drop)
        .select(pl.exclude(exclusion_regex))
        .columns
)

# >> Write the feature names and the target column name to .json
with open(CONFIG_DIR / "feature-config.json", "w") as f:
    feature_config = {
        'feature_names' : feature_names,
        'target_name' : 'DW2Y'
    }
    json.dump(feature_config, f, indent = 4)

### === Feature Filtration and Selection === ###

# >> This portion depends heavily on the clinical endpoint being predicted, and
# >> the train/test patient cohorts.
# >> Objectives:
# >> 1. Pre-compute feature means and standard deviations using training
# >>    patients as a reference.
# >> 2. Pre-compute feature selection, again based on the same patient cohort.
# >>
# >> To select features, we remove correlated features (Pearson) favoring the
# >> feature with a higher mutual info score in a correlated feature pair.

clinical_df = (
    pl.read_csv(TRAINING_DATA_DIR.parent / "clinical_deidentified_filtered.csv")
        .filter(
            (pl.col("Visit_No") == 1)
        )
)

train_pt_ids = (
    clinical_df
        .filter(pl.col("train_test_val") == 0)
        .get_column(PATIENT_ID_COL)
        .unique()
        .to_list()
)
test_pt_ids = (
    clinical_df
        .filter(pl.col("train_test_val") == 1)
        .get_column(PATIENT_ID_COL)
        .unique()
        .to_list()
)

# >> Save patient train/test split IDs
with open(CONFIG_DIR / "train-test-ids-dw2y.json", "w") as f:
    json.dump(
        {
            'train_ids' : train_pt_ids,
            'test_ids' : test_pt_ids,
        },
        f,
        indent = 4
    )

# >> Format Data for Cleaning
train_files = list(filter(lambda x: int(x.stem) in train_pt_ids, data_files))
dataset = (
    LDODataset(
        train_files,
        feature_config,
        filter_condition = (
            (pl.col("GROUP") < 10) # & (pl.col("Visit_No") == 1) TODO: VERIFY ??
        )
    )
)

tensors = []
targets = []
for idx in range(len(dataset)):
    x, y = dataset.__getitem__(idx)
    n_obj = x.shape[0]
    y = [y] * x.shape[0]

    tensors.append(x)
    targets.append(th.tensor(y))

x_data = pl.from_torch(th.cat(tensors, dim = 0), schema = feature_names)
y_data = pl.Series(th.cat(targets, dim = 0))

# >> 1. Remove zero variance features
vthresher = VarianceThreshold()
vthresher.set_output(transform = "polars")

x_data = vthresher.fit_transform(x_data)
log.info(f">>> Shape of data after variance threshold: {x_data.shape}")

# >> 2. Remove Invalid (Inf) Rows (Nuclei)
x_data = (
    x_data
        .with_columns(DW2Y = y_data)
        .with_columns(pl.all().replace({999 : None}))
        .with_columns(
            pl.when(~cs.numeric().is_infinite())
            .then(cs.numeric())
            .otherwise(None)
        )
        .drop_nulls()
)
y_data = x_data.get_column("DW2Y")
x_data = x_data.drop("DW2Y")
log.info(f">>> Shape of data after invalid value removal: {x_data.shape}")

# >> 3. Redundant (correlated) features
x_data = (
    filter_correlated_features_by_mi_score(
        x_data,
        y_data,
        corr_thresh = 0.9,
        random_state = 2380
    )
)
log.info(f">>> Shape of data after redundant feature removal: {x_data.shape}")
log.info(">>> Number of features remaining: %d", x_data.shape[1])

# >> 4. z-scale features
scaler = StandardScaler()
scaler.set_output(transform = "polars")
scaler.fit(x_data)

# >> 5. Make Filtered Feature Configuration & Save Data
joblib.dump(scaler, OUTPUT_DIR / "ldo-scaler-dw2y.joblib")
with open(CONFIG_DIR / "filtered-feature-config.json", "w") as f:
    json.dump(
        {
            "feature_names" : x_data.columns,
            "target_name" : "DW2Y"
        },
        f,
        indent = 4
    )

### === Apply Scaling and Filtering to All to Copy === ###
(TRAINING_DATA_DIR / "filtered_copy").mkdir(exist_ok = True)

for f in tqdm(data_files, total = len(data_files), desc = "Clean Dataframes"):
    # >> Load LDO features for each patient
    df = pl.read_csv(f).filter(pl.col("GROUP") < 10)

    # Capture the various endpoints here which should be carried over
    endpoint_df = (
        df.select(
            [
                # Necessary metadata
                "Visit_No", "BiopsyID",
                # Clinical endpoints
                "DW2Y", "BCR", "Progression", "Duration"
            ]
        )
    )
    # Clean dataframe rows
    df = (
        df.select(x_data.columns)
            .with_columns(pl.all().replace({999 : None}))
            .with_columns(
                pl.when(~cs.numeric().is_infinite())
                .then(cs.numeric())
                .otherwise(None)
            )
            .drop_nulls()
    )
    df = (
        pl.concat(
            [scaler.transform(df), endpoint_df],
            how = 'horizontal_extend')
    )
    df.write_csv(TRAINING_DATA_DIR / f"filtered_copy/{f.stem}.csv")


