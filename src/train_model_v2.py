"""
LinsanityML — Model Training Pipeline v2

Improvements over v1:
  1. Uses cleaned 59-feature set (removes high-null + redundant features)
  2. Time-series cross-validation (not just one train/test split)
  3. Hyperparameter tuning via RandomizedSearchCV
  4. Saves production-ready model for daily predictions

The original train_model.py is preserved as-is.
"""

import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    classification_report, confusion_matrix,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve
from scipy.stats import uniform, randint

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ─── Configuration ───────────────────────────────────────────────────

TARGET = "sprd_cvrd"
TEST_SEASON = "2024-25"

# Features to EXCLUDE (identifiers, targets, leaky, high-null, redundant)
EXCLUDE_COLS = [
    # Identifiers & target
    "game_id", "game_date", "season", "season_type",
    "home_team", "away_team",
    "home_pts", "away_pts", "actual_margin",
    "sprd_cvrd", "commence_time",
    # High-null features (incomplete data fetches)
    "attendance", "g_time_mins", "is_national_tv", "home_temp_c",
    # Redundant (perfectly correlated with kept features)
    "away_w_l_opp_last_2",      # = home_w_l_opp_last_2
    "away_w_l_opp_last_10",     # = home_w_l_opp_last_10
    "home_pts_opp_last_2",      # = away_allow_opp_last_2
    "home_pts_opp_last_10",     # = away_allow_opp_last_10
    "home_allow_opp_last_2",    # = away_pts_opp_last_2
    "home_allow_opp_last_10",   # = away_pts_opp_last_10
    "home_g_num",               # ≈ away_g_num (r=0.999)
    "home_rest_days",           # ≈ away_rest_days (r=0.952)
    "home_pace_10",             # ≈ home_allow_last_10 (r=0.856)
]


def load_data(data_dir):
    """Load and prepare the dataset."""
    path = os.path.join(data_dir, "features_with_spreads.csv")
    df = pd.read_csv(path)
    df["game_date"] = pd.to_datetime(df["game_date"])

    # Feature columns
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feature_cols].copy()

    # Include spread as feature (known pre-game)
    if "f_d_sprd_away" in df.columns and "f_d_sprd_away" not in feature_cols:
        X["f_d_sprd_away"] = df["f_d_sprd_away"]
        feature_cols.append("f_d_sprd_away")

    # Drop any remaining object columns
    obj_cols = X.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        X = X.drop(columns=obj_cols)
        feature_cols = [c for c in feature_cols if c not in obj_cols]

    y = df[TARGET].values

    print(f"Loaded {len(df)} games, {len(feature_cols)} features")
    print(f"Target rate: {np.mean(y):.3f}")

    return df, X, y, feature_cols


def time_split(df, X, y):
    """Time-based train/test split."""
    train_mask = df["season"] != TEST_SEASON
    test_mask = df["season"] == TEST_SEASON
    return (X[train_mask], X[test_mask],
            y[train_mask], y[test_mask])


def cross_validate_baseline(X_train, y_train, feature_cols):
    """
    Time-series cross-validation to get robust performance estimates.
    Uses 4 folds (each fold trains on earlier data, tests on later).
    """
    print("\n--- Time-Series Cross-Validation (4 folds) ---")
    tscv = TimeSeriesSplit(n_splits=4)

    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=300, max_depth=8, min_samples_leaf=20,
                random_state=42, n_jobs=-1)),
        ]),
        "XGBoost": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
                random_state=42, eval_metric="logloss", verbosity=0)),
        ]),
    }

    cv_results = {}
    for name, model in models.items():
        fold_scores = {"accuracy": [], "roc_auc": [], "f1": []}

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            Xt, Xv = X_train.iloc[train_idx], X_train.iloc[val_idx]
            yt, yv = y_train[train_idx], y_train[val_idx]

            model.fit(Xt[feature_cols], yt)
            y_pred = model.predict(Xv[feature_cols])
            y_prob = model.predict_proba(Xv[feature_cols])[:, 1]

            fold_scores["accuracy"].append(accuracy_score(yv, y_pred))
            fold_scores["roc_auc"].append(roc_auc_score(yv, y_prob))
            fold_scores["f1"].append(f1_score(yv, y_pred))

        cv_results[name] = {
            metric: (np.mean(scores), np.std(scores))
            for metric, scores in fold_scores.items()
        }

        acc_m, acc_s = cv_results[name]["accuracy"]
        auc_m, auc_s = cv_results[name]["roc_auc"]
        print(f"  {name:25s}  Acc: {acc_m:.4f}±{acc_s:.4f}  AUC: {auc_m:.4f}±{auc_s:.4f}")

    return cv_results


def tune_xgboost(X_train, y_train, feature_cols):
    """
    Hyperparameter tuning for XGBoost using RandomizedSearchCV
    with time-series cross-validation.
    """
    print("\n--- Hyperparameter Tuning (XGBoost) ---")
    print("  Running 50 random configurations across 4 CV folds...")

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBClassifier(
            random_state=42, eval_metric="logloss", verbosity=0,
        )),
    ])

    param_dist = {
        "model__n_estimators": randint(100, 600),
        "model__max_depth": randint(3, 10),
        "model__learning_rate": uniform(0.01, 0.2),
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.5, 0.5),
        "model__min_child_weight": randint(3, 30),
        "model__reg_alpha": uniform(0, 1.0),
        "model__reg_lambda": uniform(0.5, 2.0),
        "model__gamma": uniform(0, 0.5),
    }

    tscv = TimeSeriesSplit(n_splits=4)

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=50,
        cv=tscv,
        scoring="roc_auc",
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )

    search.fit(X_train[feature_cols], y_train)

    print(f"\n  Best CV ROC-AUC: {search.best_score_:.4f}")
    print(f"  Best parameters:")
    for param, val in search.best_params_.items():
        short = param.replace("model__", "")
        if isinstance(val, float):
            print(f"    {short:25s} {val:.4f}")
        else:
            print(f"    {short:25s} {val}")

    return search.best_estimator_, search.best_params_, search.best_score_


def tune_random_forest(X_train, y_train, feature_cols):
    """Hyperparameter tuning for Random Forest."""
    print("\n--- Hyperparameter Tuning (Random Forest) ---")
    print("  Running 30 random configurations across 4 CV folds...")

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(random_state=42, n_jobs=-1)),
    ])

    param_dist = {
        "model__n_estimators": randint(100, 600),
        "model__max_depth": randint(4, 15),
        "model__min_samples_leaf": randint(5, 50),
        "model__min_samples_split": randint(5, 30),
        "model__max_features": uniform(0.3, 0.7),
    }

    tscv = TimeSeriesSplit(n_splits=4)

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=30,
        cv=tscv,
        scoring="roc_auc",
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )

    search.fit(X_train[feature_cols], y_train)

    print(f"\n  Best CV ROC-AUC: {search.best_score_:.4f}")
    print(f"  Best parameters:")
    for param, val in search.best_params_.items():
        short = param.replace("model__", "")
        if isinstance(val, float):
            print(f"    {short:25s} {val:.4f}")
        else:
            print(f"    {short:25s} {val}")

    return search.best_estimator_, search.best_params_, search.best_score_


def final_evaluation(models, X_test, y_test, feature_cols, output_dir):
    """Evaluate all tuned models on the holdout test set."""
    print(f"\n{'='*60}")
    print(f"FINAL EVALUATION — {TEST_SEASON} holdout ({len(X_test)} games)")
    print(f"{'='*60}")

    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test[feature_cols])
        y_prob = model.predict_proba(X_test[feature_cols])[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "f1": f1_score(y_test, y_pred),
            "brier": brier_score_loss(y_test, y_prob),
        }

        results[name] = {"metrics": metrics, "y_pred": y_pred, "y_prob": y_prob}

        print(f"\n  {name}:")
        print(f"    Accuracy:  {metrics['accuracy']:.4f}")
        print(f"    ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"    F1:        {metrics['f1']:.4f}")
        print(f"    Brier:     {metrics['brier']:.4f}")
        print(classification_report(y_test, y_pred,
              target_names=["Not Covered", "Covered"], digits=4))

    # Feature importance for best model
    best_name = max(results, key=lambda m: results[m]["metrics"]["roc_auc"])
    best_model = models[best_name]

    imp = best_model.named_steps["model"].feature_importances_
    n = len(imp)
    importance = pd.DataFrame({
        "feature": feature_cols[:n],
        "importance": imp,
    }).sort_values("importance", ascending=False)

    # Plots
    os.makedirs(output_dir, exist_ok=True)

    # Model comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    model_names = list(results.keys())
    metrics_list = ["accuracy", "roc_auc", "f1"]
    x = np.arange(len(model_names))
    width = 0.25
    for i, metric in enumerate(metrics_list):
        values = [results[m]["metrics"][metric] for m in model_names]
        ax.bar(x + i * width, values, width, label=metric.upper())
    ax.set_xticks(x + width)
    ax.set_xticklabels(model_names, rotation=15)
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5)
    ax.set_ylim(0.4, 0.75)
    ax.set_title("Model Comparison (v2 — cleaned features + tuned hyperparameters)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison_v2.png"), dpi=150)
    plt.close()

    # Feature importance
    fig, ax = plt.subplots(figsize=(10, 8))
    top20 = importance.head(20)
    ax.barh(range(len(top20)), top20["importance"].values, color="steelblue")
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20["feature"].values)
    ax.invert_yaxis()
    ax.set_title(f"Top 20 Features — {best_name} (tuned)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance_v2.png"), dpi=150)
    plt.close()

    # Calibration
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    for name, res in results.items():
        prob_true, prob_pred = calibration_curve(y_test, res["y_prob"], n_bins=10)
        ax.plot(prob_pred, prob_true, marker="o", label=name)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curves (v2)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "calibration_v2.png"), dpi=150)
    plt.close()

    # Confusion matrices
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
    plt.savefig(os.path.join(output_dir, "confusion_matrices_v2.png"), dpi=150)
    plt.close()

    importance.to_csv(os.path.join(output_dir, "feature_importance_v2.csv"), index=False)

    return results, best_name, importance


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(model_dir, exist_ok=True)

    print("=" * 60)
    print("LinsanityML — Model Training v2")
    print("  (cleaned features + cross-validation + tuning)")
    print("=" * 60)

    # Load
    df, X, y, feature_cols = load_data(data_dir)
    X_train, X_test, y_train, y_test = time_split(df, X, y)
    print(f"Train: {len(X_train)} | Test: {len(X_test)} ({TEST_SEASON})")

    # Step 1: Cross-validation baselines
    cv_results = cross_validate_baseline(X_train, y_train, feature_cols)

    # Step 2: Hyperparameter tuning
    best_xgb, xgb_params, xgb_cv_score = tune_xgboost(X_train, y_train, feature_cols)
    best_rf, rf_params, rf_cv_score = tune_random_forest(X_train, y_train, feature_cols)

    # Also train a tuned logistic regression (less to tune)
    best_lr = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, C=0.5, random_state=42)),
    ])
    best_lr.fit(X_train[feature_cols], y_train)

    # Step 3: Final evaluation on holdout
    tuned_models = {
        "Logistic Regression": best_lr,
        "Random Forest (tuned)": best_rf,
        "XGBoost (tuned)": best_xgb,
    }

    results, best_name, importance = final_evaluation(
        tuned_models, X_test, y_test, feature_cols, output_dir
    )

    # Save best model
    best_metrics = results[best_name]["metrics"]
    model_path = os.path.join(model_dir, "best_model_v2.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": tuned_models[best_name],
            "feature_cols": feature_cols,
            "model_name": best_name,
            "metrics": best_metrics,
            "test_season": TEST_SEASON,
            "version": "v2",
        }, f)

    # Summary
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE — v2 SUMMARY")
    print(f"{'='*60}")
    print(f"  Features: {len(feature_cols)} (cleaned from 74)")
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"\n  Cross-Validation Results (4-fold time-series):")
    for name, res in cv_results.items():
        acc_m, acc_s = res["accuracy"]
        auc_m, auc_s = res["roc_auc"]
        print(f"    {name:25s}  Acc: {acc_m:.4f}±{acc_s:.4f}  AUC: {auc_m:.4f}±{auc_s:.4f}")

    print(f"\n  Tuned Model Results ({TEST_SEASON} holdout):")
    print(f"  {'Model':30s} {'Accuracy':>10s} {'ROC-AUC':>10s} {'F1':>10s}")
    print(f"  {'-'*60}")
    for name, res in results.items():
        m = res["metrics"]
        marker = " ← BEST" if name == best_name else ""
        print(f"  {name:30s} {m['accuracy']:>10.4f} {m['roc_auc']:>10.4f} {m['f1']:>10.4f}{marker}")
    print(f"\n  Coin flip baseline: {max(np.mean(y_test), 1-np.mean(y_test)):.4f}")
    print(f"  Best model: {best_name}")
    print(f"  Saved: {model_path}")
    print(f"{'='*60}")

    # Save summary JSON
    summary = {
        "version": "v2",
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "excluded_cols": EXCLUDE_COLS,
        "test_season": TEST_SEASON,
        "cv_results": {
            name: {metric: {"mean": round(m[0], 4), "std": round(m[1], 4)}
                   for metric, m in res.items()}
            for name, res in cv_results.items()
        },
        "holdout_results": {
            name: {k: round(v, 4) for k, v in res["metrics"].items()}
            for name, res in results.items()
        },
        "best_model": best_name,
        "best_xgb_params": {k.replace("model__", ""): (round(v, 4) if isinstance(v, float) else v)
                            for k, v in xgb_params.items()},
    }
    with open(os.path.join(output_dir, "training_summary_v2.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
