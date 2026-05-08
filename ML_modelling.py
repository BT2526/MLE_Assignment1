"""
ML_modelling.py
---------------
Logistic Regression on Gold Tables (Baseline & Full feature sets).

Pipeline:
  1. Load gold feature tables (baseline or full) + label store
  2. Join on Customer_ID; time-based train / test split
  3. Impute missing values, scale features
  4. Fit Logistic Regression
  5. Evaluate: Accuracy, ROC-AUC, Classification Report, Confusion Matrix
  6. Save outputs (metrics CSV + confusion matrix PNG) to datamart/outputs/

Usage:
  python ML_modelling.py                         # baseline features (default)
  python ML_modelling.py --feature_set full      # full features (with clickstream)
  python ML_modelling.py --feature_set baseline --train_end 2024-06-01
"""

import os
import glob
import argparse
import warnings
from datetime import datetime

# Suppress transient numpy overflow warnings from lbfgs intermediate steps
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*matmul.*")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # headless-safe backend

import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR            = os.path.dirname(os.path.abspath(__file__))
GOLD_LABEL_DIR      = os.path.join(BASE_DIR, "datamart/gold/label_store")
GOLD_BASELINE_DIR   = os.path.join(BASE_DIR, "datamart/gold/features/baseline")
GOLD_FULL_DIR       = os.path.join(BASE_DIR, "datamart/gold/features/full")
OUTPUT_DIR          = os.path.join(BASE_DIR, "datamart/outputs")

# Columns never used as model features
NON_FEATURE_COLS = {"Customer_ID", "loan_id", "snapshot_date", "label", "label_def"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_parquet_dir(spark, directory: str) -> pyspark.sql.DataFrame:
    """Read all parquet partitions under a directory into a single Spark DF."""
    files = sorted(glob.glob(os.path.join(directory, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files found in: {directory}")
    return spark.read.option("header", "true").parquet(*files)


def time_split(df: pd.DataFrame, train_end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-based train / test split.
    Rows with snapshot_date <= train_end go to train, the rest to test.

    Parameters
    ----------
    df         : combined pandas DataFrame with 'snapshot_date' column
    train_end  : ISO date string, e.g. '2024-06-01'
    """
    cutoff = pd.Timestamp(train_end)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    train = df[df["snapshot_date"] <= cutoff].copy()
    test  = df[df["snapshot_date"] >  cutoff].copy()
    return train, test


class PercentileClipper(BaseEstimator, TransformerMixin):
    """
    Clamps each feature column to [P1, P99] bounds learned ONLY from training data.

    Why this matters
    ----------------
    The gold-table pipeline clamps outliers per snapshot (once per monthly
    parquet) using that snapshot's own P1/P99. This means training snapshots
    and test snapshots each got clamped to *different* bounds — a mild form
    of data leakage where test-set statistics influence test-set features.

    By adding this transformer as the first step in the sklearn Pipeline,
    we fit the bounds exclusively on X_train (via .fit()), then apply the
    same frozen bounds to X_test (via .transform()). The test set never
    influences its own outlier treatment.
    """
    def __init__(self, lower_pct: float = 1.0, upper_pct: float = 99.0):
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct

    def fit(self, X, y=None):
        # Compute bounds from training data only
        self.lower_ = np.nanpercentile(X, self.lower_pct, axis=0)  # shape: (n_features,)
        self.upper_ = np.nanpercentile(X, self.upper_pct, axis=0)  # shape: (n_features,)
        return self

    def transform(self, X, y=None):
        # Apply the frozen training bounds to whatever data is passed in
        return np.clip(X, self.lower_, self.upper_)


def build_sklearn_pipeline(max_iter: int = 1000) -> Pipeline:
    """Clip outliers (train bounds) → Impute → Scale → Logistic Regression."""
    return Pipeline([
        ("clipper", PercentileClipper(lower_pct=1.0, upper_pct=99.0)),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     LogisticRegression(
            solver="lbfgs",
            max_iter=max_iter,
            class_weight="balanced",   # handles label imbalance
            random_state=42,
        )),
    ])


def evaluate(y_true, y_pred, y_prob, label: str, output_dir: str) -> dict:
    """Compute metrics, print them, save confusion matrix, return metrics dict."""
    acc     = accuracy_score(y_true, y_pred)
    auc     = roc_auc_score(y_true, y_prob)
    report  = classification_report(y_true, y_pred, digits=4)
    cm      = confusion_matrix(y_true, y_pred)

    print(f"\n{'='*60}")
    print(f"  Evaluation — {label}")
    print(f"{'='*60}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"\n  Classification Report:\n{report}")
    print(f"  Confusion Matrix:\n{cm}\n")

    # --- Plot confusion matrix -----------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Default (0)", "Default (1)"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {label}", fontsize=11)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    cm_path = os.path.join(output_dir, f"confusion_matrix_{label.lower().replace(' ', '_')}.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved → {cm_path}")

    return {
        "label":    label,
        "accuracy": round(acc, 4),
        "roc_auc":  round(auc, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(feature_set: str = "baseline", train_end: str = "2024-06-01", max_iter: int = 1000):

    print(f"\n{'='*60}")
    print(f"  ML_modelling.py — Logistic Regression")
    print(f"  Feature set : {feature_set}")
    print(f"  Train end   : {train_end}  |  Test start: after {train_end}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # 1. Initialise Spark
    # ------------------------------------------------------------------
    spark = (
        pyspark.sql.SparkSession.builder
        .appName("ML_modelling")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    # ------------------------------------------------------------------
    # 2. Load gold tables
    # ------------------------------------------------------------------
    print("[1/5] Loading label store …")
    df_labels = load_parquet_dir(spark, GOLD_LABEL_DIR)
    print(f"      Label store: {df_labels.count():,} rows | columns: {df_labels.columns}")

    feat_dir = GOLD_BASELINE_DIR if feature_set == "baseline" else GOLD_FULL_DIR
    print(f"\n[2/5] Loading {feature_set} feature store from {feat_dir} …")
    df_features = load_parquet_dir(spark, feat_dir)
    print(f"      Features: {df_features.count():,} rows | {len(df_features.columns)} columns")

    # ------------------------------------------------------------------
    # 3. Join features + labels
    # ------------------------------------------------------------------
    print("\n[3/5] Joining features ↔ labels …")
    #
    # Key insight:
    #   • label store snapshot_date  = observation date (loan_open + mob months)
    #   • feature store snapshot_date = loan origination / feature capture date
    #
    # loan_id format: CUS_0x<id>_YYYY_MM_DD  → loan opened YYYY-MM-DD
    # So we join on:  Customer_ID  +  loan_open_date == feature snapshot_date

    # Extract loan_open_date from loan_id (last three underscore-split parts = YYYY_MM_DD)
    # Note: Customer_ID is already lowercased at the silver layer (data_processing_silver_table.py)
    df_labels_prep = df_labels \
        .withColumn("label_obs_date", F.col("snapshot_date").cast("string")) \
        .withColumn("loan_open_date",
            F.concat_ws("-",
                F.split(F.col("loan_id"), "_")[2],
                F.split(F.col("loan_id"), "_")[3],
                F.split(F.col("loan_id"), "_")[4],
            )
        ) \
        .select("Customer_ID", "loan_open_date", "label_obs_date", "label")

    print(f"      Feature store sample: {df_features.select('Customer_ID','snapshot_date').first()}")
    print(f"      Label store sample  : {df_labels_prep.select('Customer_ID','loan_open_date','label_obs_date').first()}")

    df_merged = df_features.join(
        df_labels_prep,
        on=[
            df_features["Customer_ID"] == df_labels_prep["Customer_ID"],
            df_features["snapshot_date"] == df_labels_prep["loan_open_date"],
        ],
        how="inner",
    ).drop(df_labels_prep["Customer_ID"])   # keep features Customer_ID

    total_rows = df_merged.count()
    print(f"      Merged dataset: {total_rows:,} rows | {len(df_merged.columns)} columns")

    if total_rows == 0:
        raise ValueError(
            "Join produced 0 rows. "
            "Verify that Customer_ID values and loan origination dates overlap "
            "between the feature store and the label store."
        )

    # Label distribution
    label_dist = df_merged.groupBy("label").count().orderBy("label").toPandas()
    print("\n      Label distribution:")
    for _, row in label_dist.iterrows():
        pct = row["count"] / total_rows * 100
        print(f"        label={int(row['label'])}: {int(row['count']):,} ({pct:.1f}%)")

    # ------------------------------------------------------------------
    # 4. Convert to Pandas; train / test split
    # ------------------------------------------------------------------
    # Train/test split is based on label_obs_date (the observation date when
    # the label is determined), which represents true chronological ordering.
    print("\n[4/5] Converting to Pandas & splitting …")
    df_pd = df_merged.toPandas()

    # Use label_obs_date for the time-based split
    df_pd["label_obs_date"] = pd.to_datetime(df_pd["label_obs_date"])
    cutoff = pd.Timestamp(train_end)
    train_df = df_pd[df_pd["label_obs_date"] <= cutoff].copy()
    test_df  = df_pd[df_pd["label_obs_date"] >  cutoff].copy()

    print(f"      Train: {len(train_df):,} rows (label_obs_date ≤ {train_end})")
    print(f"      Test : {len(test_df):,} rows  (label_obs_date >  {train_end})")

    if len(test_df) == 0:
        raise ValueError(
            f"Test set is empty. All label_obs_dates fall on or before train_end={train_end}. "
            "Try a smaller train_end date, e.g. '2024-06-01'."
        )

    # Columns never used as model features (add loan_open_date & label_obs_date)
    _non_feat = NON_FEATURE_COLS | {"loan_open_date", "label_obs_date"}
    feature_cols = [c for c in df_pd.columns if c not in _non_feat]
    print(f"\n      Feature columns ({len(feature_cols)}): {feature_cols}")

    X_train = train_df[feature_cols].astype(float)
    y_train = train_df["label"].astype(int)
    X_test  = test_df[feature_cols].astype(float)
    y_test  = test_df["label"].astype(int)

    # ------------------------------------------------------------------
    # 5. Train Logistic Regression
    # ------------------------------------------------------------------
    print(f"\n[5/5] Training Logistic Regression (max_iter={max_iter}) …")
    pipe = build_sklearn_pipeline(max_iter=max_iter)
    pipe.fit(X_train, y_train)
    print("      Training complete.")

    # ------------------------------------------------------------------
    # 6. Evaluate
    # ------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_label = f"LR_{feature_set}"
    results = []

    # Train performance
    y_train_pred = pipe.predict(X_train)
    y_train_prob = pipe.predict_proba(X_train)[:, 1]
    results.append(evaluate(y_train, y_train_pred, y_train_prob,
                            label=f"{run_label}_train", output_dir=OUTPUT_DIR))

    # Test performance
    y_test_pred = pipe.predict(X_test)
    y_test_prob = pipe.predict_proba(X_test)[:, 1]
    results.append(evaluate(y_test, y_test_pred, y_test_prob,
                            label=f"{run_label}_test",  output_dir=OUTPUT_DIR))

    # ------------------------------------------------------------------
    # 7. Coefficient table (top features)
    # ------------------------------------------------------------------
    lr_model   = pipe.named_steps["clf"]
    coef_series = pd.Series(lr_model.coef_[0], index=feature_cols)
    coef_df = (
        coef_series
        .abs()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "feature", 0: "abs_coef"})
    )
    coef_df["coef"] = coef_series[coef_df["feature"]].values

    print("\n  Top-20 features by |coefficient|:")
    print(coef_df.head(20).to_string(index=False))

    coef_path = os.path.join(OUTPUT_DIR, f"coefficients_{run_label}.csv")
    coef_df.to_csv(coef_path, index=False)
    print(f"\n  Coefficients saved → {coef_path}")

    # Bar chart of top-20 coefficients
    fig, ax = plt.subplots(figsize=(9, 6))
    top20 = coef_df.head(20)
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in top20["coef"]]
    ax.barh(top20["feature"][::-1], top20["coef"][::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient value")
    ax.set_title(f"Top-20 LR Coefficients — {feature_set} feature set", fontsize=12)
    plt.tight_layout()
    coef_plot_path = os.path.join(OUTPUT_DIR, f"coef_plot_{run_label}.png")
    fig.savefig(coef_plot_path, dpi=150)
    plt.close(fig)
    print(f"  Coefficient plot saved → {coef_plot_path}")

    # ------------------------------------------------------------------
    # 8. Save summary metrics CSV
    # ------------------------------------------------------------------
    metrics_df = pd.DataFrame(results)
    metrics_df["feature_set"] = feature_set
    metrics_df["train_end"]   = train_end
    metrics_df["run_ts"]      = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metrics_path = os.path.join(OUTPUT_DIR, f"metrics_{run_label}.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n  Summary metrics saved → {metrics_path}")
    print(f"\n  {metrics_df.to_string(index=False)}")

    spark.stop()
    print("\n  Done.\n")

    return pipe, metrics_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Logistic Regression on Gold Tables")
    parser.add_argument(
        "--feature_set",
        choices=["baseline", "full"],
        default="baseline",
        help="Which gold feature table to use: 'baseline' (financials + attributes) "
             "or 'full' (+ clickstream fe_1..fe_20). Default: baseline",
    )
    parser.add_argument(
        "--train_end",
        default="2024-06-01",
        help="ISO date (YYYY-MM-DD). Snapshots up to and including this date form "
             "the training set; later snapshots form the test set. Default: 2024-06-01",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=1000,
        help="Maximum iterations for the LR solver. Default: 1000",
    )
    args = parser.parse_args()
    main(feature_set=args.feature_set, train_end=args.train_end, max_iter=args.max_iter)
