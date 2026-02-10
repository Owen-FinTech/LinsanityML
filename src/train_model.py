"""
LinsanityML — Model Training Pipeline

Trains and evaluates models to predict whether the away team covers
the FanDuel spread (binary classification: sprd_cvrd = 1 or 0).

Approach:
  1. Time-based train/test split (no data leakage from future)
  2. Three models: Logistic Regression (baseline), Random Forest, XGBoost
  3. Evaluation: accuracy, ROC-AUC, precision/recall, calibration
  4. Feature importance analysis
  5. Saves best model for inference

Key ML concepts (for Owen's learning):
  - We split by TIME, not randomly, because in production we'd always be
    predicting future games using past data
  - ROC-AUC measures how well we rank predictions (>0.5 = better than coin flip)
  - Calibration tells us if "70% confidence" really means 70% of the time
  - Feature importance tells us what the model actually uses
"""

import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Warning: xgboost not installed, skipping XGBoost model")


# ─── Configuration ───────────────────────────────────────────────────

# Features to EXCLUDE from training (identifiers, targets, leaky columns)
EXCLUDE_COLS = [
    "game_id", "game_date", "season", "season_type",
    "home_team", "away_team",
    "home_pts", "away_pts", "actual_margin",  # These ARE the outcome
    "sprd_cvrd",        # This is the TARGET
    "f_d_sprd_away",    # Include spread as a feature (it's known pre-game)
    "commence_time",    # Metadata
]

# Target column
TARGET = "sprd_cvrd"

# Train/test split: use last season as test set
# This simulates "train on history, predict current season"
TEST_SEASON = "2024-25"


def load_data(data_dir):
    """Load the features + spreads dataset."""
    path = os.path.join(data_dir, "features_with_spreads.csv")
    if not os.path.exists(path):
        print("ERROR: features_with_spreads.csv not found!")
        print("Run build_features.py then merge with spreads first.")
        sys.exit(1)

    df = pd.read_csv(path)
    df["game_date"] = pd.to_datetime(df["game_date"])
    print(f"Loaded {len(df)} games with {df.shape[1]} columns")
    return df


def prepare_features(df):
    """
    Prepare feature matrix X and target y.
    Handles column selection, ensures spread is included as a feature.
    """
    # The spread IS a pre-game feature (we know it before tip-off)
    # So we include it in training — it's the line we're trying to beat

    # Identify feature columns (everything not excluded, except keep spread)
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]

    # But we DO want the spread as a feature
    if "f_d_sprd_away" in df.columns and "f_d_sprd_away" not in feature_cols:
        feature_cols.append("f_d_sprd_away")

    # Drop any remaining string/object columns
    X = df[feature_cols].copy()
    obj_cols = X.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        print(f"  Dropping non-numeric columns: {obj_cols}")
        X = X.drop(columns=obj_cols)

    feature_cols = X.columns.tolist()
    y = df[TARGET].values

    print(f"  Features: {len(feature_cols)}")
    print(f"  Target distribution: {np.mean(y):.3f} (spread covered rate)")

    return X, y, feature_cols


def time_split(df, X, y, test_season=TEST_SEASON):
    """
    Split data by season (time-based, no leakage).
    Train = all seasons before test_season
    Test = test_season only
    """
    train_mask = df["season"] != test_season
    test_mask = df["season"] == test_season

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    print(f"\n  Time-based split:")
    print(f"    Train: {len(X_train)} games (seasons before {test_season})")
    print(f"    Test:  {len(X_test)} games ({test_season})")
    print(f"    Train target rate: {np.mean(y_train):.3f}")
    print(f"    Test target rate:  {np.mean(y_test):.3f}")

    return X_train, X_test, y_train, y_test


def build_models():
    """Build the three model pipelines."""
    models = {}

    # 1. Logistic Regression (baseline)
    # Pipeline: impute NaN → scale features → logistic regression
    models["Logistic Regression"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            C=1.0,           # Regularization (1.0 = moderate)
            solver="lbfgs",
            random_state=42,
        )),
    ])

    # 2. Random Forest
    # Tree-based models handle NaN via imputation, don't need scaling
    models["Random Forest"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=300,
            max_depth=8,          # Limit depth to prevent overfitting
            min_samples_leaf=20,  # Require 20 samples per leaf
            random_state=42,
            n_jobs=-1,
        )),
    ])

    # 3. XGBoost
    if HAS_XGB:
        models["XGBoost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=10,
                reg_alpha=0.1,        # L1 regularization
                reg_lambda=1.0,       # L2 regularization
                random_state=42,
                eval_metric="logloss",
                verbosity=0,
            )),
        ])

    return models


def evaluate_model(name, model, X_test, y_test):
    """Evaluate a trained model and return metrics dict."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "brier": brier_score_loss(y_test, y_prob),
    }

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}  (baseline: {max(np.mean(y_test), 1-np.mean(y_test)):.4f})")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}  (0.5 = random, 1.0 = perfect)")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  Brier:     {metrics['brier']:.4f}  (lower = better calibrated)")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Not Covered", "Covered"]))

    return metrics, y_pred, y_prob


def get_feature_importance(model, feature_cols, name):
    """Extract feature importance from trained model."""
    # After imputation, feature count should match but let's be safe
    if "Logistic" in name:
        coefs = model.named_steps["model"].coef_[0]
        n = len(coefs)
        cols = feature_cols[:n] if len(feature_cols) >= n else feature_cols + [f"feat_{i}" for i in range(len(feature_cols), n)]
        importance = pd.DataFrame({
            "feature": cols[:n],
            "importance": np.abs(coefs),
            "coefficient": coefs,
        }).sort_values("importance", ascending=False)
    else:  # Random Forest or XGBoost
        imp = model.named_steps["model"].feature_importances_
        n = len(imp)
        cols = feature_cols[:n] if len(feature_cols) >= n else feature_cols + [f"feat_{i}" for i in range(len(feature_cols), n)]
        importance = pd.DataFrame({
            "feature": cols[:n],
            "importance": imp,
        }).sort_values("importance", ascending=False)

    return importance


def plot_results(results, y_test, output_dir):
    """Generate evaluation plots."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Model comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    model_names = list(results.keys())
    metrics_to_plot = ["accuracy", "roc_auc", "f1"]
    x = np.arange(len(model_names))
    width = 0.25

    for i, metric in enumerate(metrics_to_plot):
        values = [results[m]["metrics"][metric] for m in model_names]
        ax.bar(x + i * width, values, width, label=metric.upper())

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.set_xticks(x + width)
    ax.set_xticklabels(model_names, rotation=15)
    ax.legend()
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Random baseline")
    ax.set_ylim(0.4, 0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison.png"), dpi=150)
    plt.close()

    # 2. Feature importance (top 20) for best model
    best_name = max(results, key=lambda m: results[m]["metrics"]["roc_auc"])
    best_imp = results[best_name]["importance"]

    fig, ax = plt.subplots(figsize=(10, 8))
    top20 = best_imp.head(20)
    ax.barh(range(len(top20)), top20["importance"].values, color="steelblue")
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20["feature"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(f"Top 20 Features — {best_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=150)
    plt.close()

    # 3. Calibration curves
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")

    for name, res in results.items():
        prob_true, prob_pred = calibration_curve(y_test, res["y_prob"], n_bins=10)
        ax.plot(prob_pred, prob_true, marker="o", label=name)

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curves")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "calibration.png"), dpi=150)
    plt.close()

    # 4. Confusion matrices
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    if len(results) == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, res["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Not Covered", "Covered"],
                    yticklabels=["Not Covered", "Covered"])
        ax.set_title(name)
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrices.png"), dpi=150)
    plt.close()

    print(f"\n  Plots saved to {output_dir}/")
    return best_name


def main():
    warnings.filterwarnings("ignore")

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # ─── Load & Prepare ──────────────────────────────────────────────
    print("=" * 60)
    print("LinsanityML — Model Training")
    print("=" * 60)

    df = load_data(data_dir)
    X, y, feature_cols = prepare_features(df)
    X_train, X_test, y_train, y_test = time_split(df, X, y)

    # ─── Train Models ────────────────────────────────────────────────
    print("\nTraining models...")
    models = build_models()
    results = {}

    for name, model in models.items():
        print(f"\n  Training {name}...")
        model.fit(X_train, y_train)

        metrics, y_pred, y_prob = evaluate_model(name, model, X_test, y_test)
        importance = get_feature_importance(model, feature_cols, name)

        results[name] = {
            "model": model,
            "metrics": metrics,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "importance": importance,
        }

        print(f"  Top 10 features:")
        for _, row in importance.head(10).iterrows():
            print(f"    {row['feature']:35s} {row['importance']:.4f}")

    # ─── Plots ────────────────────────────────────────────────────────
    print("\nGenerating evaluation plots...")
    best_name = plot_results(results, y_test, output_dir)

    # ─── Save Best Model ──────────────────────────────────────────────
    print(f"\n  Best model by ROC-AUC: {best_name}")
    best_model = results[best_name]["model"]
    model_path = os.path.join(model_dir, "best_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": best_model,
            "feature_cols": feature_cols,
            "model_name": best_name,
            "metrics": results[best_name]["metrics"],
            "test_season": TEST_SEASON,
        }, f)
    print(f"  Saved: {model_path}")

    # ─── Save Feature Importance ──────────────────────────────────────
    for name, res in results.items():
        safe_name = name.lower().replace(" ", "_")
        res["importance"].to_csv(
            os.path.join(output_dir, f"feature_importance_{safe_name}.csv"),
            index=False,
        )

    # ─── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE — SUMMARY")
    print(f"{'='*60}")
    print(f"  Training games: {len(X_train)}")
    print(f"  Test games:     {len(X_test)} ({TEST_SEASON})")
    print(f"  Features:       {len(feature_cols)}")
    print(f"\n  Model Results (on {TEST_SEASON} test set):")
    print(f"  {'Model':25s} {'Accuracy':>10s} {'ROC-AUC':>10s} {'F1':>10s}")
    print(f"  {'-'*55}")
    for name, res in results.items():
        m = res["metrics"]
        print(f"  {name:25s} {m['accuracy']:>10.4f} {m['roc_auc']:>10.4f} {m['f1']:>10.4f}")
    print(f"\n  Coin flip baseline:       {max(np.mean(y_test), 1-np.mean(y_test)):.4f}")
    print(f"  Best model: {best_name}")
    print(f"\n  Outputs: {output_dir}/")
    print(f"  Model:   {model_path}")
    print(f"{'='*60}")

    # Save summary as JSON
    summary = {
        "test_season": TEST_SEASON,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "n_features": len(feature_cols),
        "results": {
            name: {k: round(v, 4) for k, v in res["metrics"].items()}
            for name, res in results.items()
        },
        "best_model": best_name,
        "feature_cols": feature_cols,
    }
    with open(os.path.join(output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
