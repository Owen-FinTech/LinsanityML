"""
LinsanityML — Feature Selection Analysis

Analyses which features contribute to model performance and which can be cut.
Produces:
  1. Correlation analysis (identify redundant features)
  2. Feature importance ranking from XGBoost
  3. Null analysis (features with too many missing values)
  4. Trains models with reduced feature sets and compares to baseline
  5. Saves recommended feature list

Does NOT modify existing files — saves results to output/feature_selection/
"""

import os
import sys
import json
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
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings("ignore")

# Same exclusions as train_model.py
EXCLUDE_COLS = [
    "game_id", "game_date", "season", "season_type",
    "home_team", "away_team",
    "home_pts", "away_pts", "actual_margin",
    "sprd_cvrd", "f_d_sprd_away", "commence_time",
]
TARGET = "sprd_cvrd"
TEST_SEASON = "2024-25"


def load_and_split(data_dir):
    """Load data and do time-based split."""
    df = pd.read_csv(os.path.join(data_dir, "features_with_spreads.csv"))
    df["game_date"] = pd.to_datetime(df["game_date"])

    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feature_cols].copy()
    obj_cols = X.select_dtypes(include=["object"]).columns.tolist()
    X = X.drop(columns=obj_cols)
    feature_cols = X.columns.tolist()

    if "f_d_sprd_away" in df.columns:
        X["f_d_sprd_away"] = df["f_d_sprd_away"]
        if "f_d_sprd_away" not in feature_cols:
            feature_cols.append("f_d_sprd_away")

    y = df[TARGET].values
    train_mask = df["season"] != TEST_SEASON
    test_mask = df["season"] == TEST_SEASON

    return (X[train_mask], X[test_mask], y[train_mask], y[test_mask],
            feature_cols, df)


def null_analysis(X_train, X_test, feature_cols):
    """Identify features with high null rates."""
    full = pd.concat([X_train, X_test])
    null_rates = full[feature_cols].isnull().mean().sort_values(ascending=False)
    high_null = null_rates[null_rates > 0.3]

    print("\n--- Null Analysis ---")
    print(f"Features with >30% nulls ({len(high_null)}):")
    for feat, rate in high_null.items():
        print(f"  {feat:40s} {rate:.1%} null")

    return high_null.index.tolist()


def correlation_analysis(X_train, feature_cols, output_dir):
    """Find highly correlated feature pairs."""
    # Drop columns that are entirely null, then impute the rest
    non_null_cols = [c for c in feature_cols if X_train[c].notna().any()]
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(
        imputer.fit_transform(X_train[non_null_cols]),
        columns=non_null_cols
    )

    corr = X_imp.corr().abs()
    corr_cols = corr.columns.tolist()

    # Find pairs with correlation > 0.85
    high_corr_pairs = []
    for i in range(len(corr_cols)):
        for j in range(i + 1, len(corr_cols)):
            if corr.iloc[i, j] > 0.85:
                high_corr_pairs.append((
                    corr_cols[i], corr_cols[j], corr.iloc[i, j]
                ))

    high_corr_pairs.sort(key=lambda x: -x[2])

    print(f"\n--- Correlation Analysis ---")
    print(f"Highly correlated pairs (r > 0.85): {len(high_corr_pairs)}")
    for f1, f2, r in high_corr_pairs[:20]:
        print(f"  {f1:35s} ↔ {f2:35s}  r={r:.3f}")

    # Plot correlation heatmap (top 30 features by variance)
    top_var = X_imp.var().sort_values(ascending=False).head(30).index
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(X_imp[top_var].corr(), annot=False, cmap="RdBu_r",
                center=0, ax=ax, square=True)
    ax.set_title("Feature Correlation Matrix (top 30 by variance)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_matrix.png"), dpi=150)
    plt.close()

    return high_corr_pairs


def get_importance_ranking(X_train, y_train, feature_cols):
    """Train XGBoost and get feature importance ranking."""
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42,
            eval_metric="logloss", verbosity=0,
        )),
    ])
    pipe.fit(X_train[feature_cols], y_train)

    imp = pipe.named_steps["model"].feature_importances_
    ranking = pd.DataFrame({
        "feature": feature_cols[:len(imp)],
        "importance": imp,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return ranking, pipe


def train_with_features(X_train, X_test, y_train, y_test, features, name=""):
    """Train XGBoost with a specific feature set and return metrics."""
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42,
            eval_metric="logloss", verbosity=0,
        )),
    ])
    pipe.fit(X_train[features], y_train)
    y_prob = pipe.predict_proba(X_test[features])[:, 1]
    y_pred = pipe.predict(X_test[features])

    return {
        "name": name,
        "n_features": len(features),
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "f1": f1_score(y_test, y_pred),
    }


def select_features(ranking, high_null_features, high_corr_pairs):
    """
    Build recommended feature sets by removing:
    1. Features with >30% nulls
    2. One from each highly correlated pair (keep the more important one)
    3. Bottom N features by importance
    """
    # Start with all features ranked by importance
    all_feats = ranking["feature"].tolist()
    importance_map = dict(zip(ranking["feature"], ranking["importance"]))

    # Mark features to remove
    to_remove = set()

    # Remove high-null features
    for f in high_null_features:
        if f in importance_map:
            to_remove.add(f)

    # From correlated pairs, remove the less important one
    for f1, f2, r in high_corr_pairs:
        imp1 = importance_map.get(f1, 0)
        imp2 = importance_map.get(f2, 0)
        weaker = f2 if imp1 >= imp2 else f1
        to_remove.add(weaker)

    # Build feature sets
    cleaned = [f for f in all_feats if f not in to_remove]
    top_30 = cleaned[:30]
    top_20 = cleaned[:20]

    return {
        "all_74": all_feats,
        "cleaned": cleaned,
        "top_30": top_30,
        "top_20": top_20,
        "removed": sorted(to_remove),
    }


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    output_dir = os.path.join(os.path.dirname(__file__), "..", "output", "feature_selection")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("LinsanityML — Feature Selection Analysis")
    print("=" * 60)

    # Load and split
    X_train, X_test, y_train, y_test, feature_cols, df = load_and_split(data_dir)
    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # 1. Null analysis
    high_null = null_analysis(X_train, X_test, feature_cols)

    # 2. Correlation analysis
    high_corr = correlation_analysis(X_train, feature_cols, output_dir)

    # 3. Feature importance ranking
    print("\n--- Feature Importance (XGBoost) ---")
    ranking, _ = get_importance_ranking(X_train, y_train, feature_cols)
    print("\nFull ranking:")
    for i, row in ranking.iterrows():
        marker = " ← HIGH NULL" if row["feature"] in high_null else ""
        print(f"  {i+1:3d}. {row['feature']:40s} {row['importance']:.4f}{marker}")

    ranking.to_csv(os.path.join(output_dir, "importance_ranking.csv"), index=False)

    # 4. Build feature sets
    feature_sets = select_features(ranking, high_null, high_corr)

    print(f"\n--- Feature Set Comparison ---")
    print(f"Removed features ({len(feature_sets['removed'])}):")
    for f in feature_sets["removed"]:
        reason = "high null" if f in high_null else "correlated"
        print(f"  ✂ {f:40s} ({reason})")

    print(f"\nFeature set sizes:")
    print(f"  All:     {len(feature_sets['all_74'])}")
    print(f"  Cleaned: {len(feature_sets['cleaned'])}")
    print(f"  Top 30:  {len(feature_sets['top_30'])}")
    print(f"  Top 20:  {len(feature_sets['top_20'])}")

    # 5. Train and compare
    print("\n--- Training with different feature sets ---")
    comparisons = []

    for set_name, feats in feature_sets.items():
        if not feats:
            continue
        result = train_with_features(X_train, X_test, y_train, y_test, feats, set_name)
        comparisons.append(result)
        print(f"  {set_name:12s} ({result['n_features']:2d} feats): "
              f"Acc={result['accuracy']:.4f}  AUC={result['roc_auc']:.4f}  F1={result['f1']:.4f}")

    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    names = [c["name"] for c in comparisons]
    accs = [c["accuracy"] for c in comparisons]
    aucs = [c["roc_auc"] for c in comparisons]
    x = np.arange(len(names))
    ax.bar(x - 0.15, accs, 0.3, label="Accuracy", color="steelblue")
    ax.bar(x + 0.15, aucs, 0.3, label="ROC-AUC", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n({comparisons[i]['n_features']})" for i, n in enumerate(names)], fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Feature Set Comparison — XGBoost")
    ax.legend()
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylim(0.45, 0.75)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_set_comparison.png"), dpi=150)
    plt.close()

    # Save recommended features
    best = max(comparisons, key=lambda c: c["roc_auc"])
    best_set = feature_sets[best["name"]]

    recommendation = {
        "recommended_set": best["name"],
        "n_features": best["n_features"],
        "accuracy": round(best["accuracy"], 4),
        "roc_auc": round(best["roc_auc"], 4),
        "features": best_set,
        "removed_features": feature_sets["removed"],
        "all_comparisons": [{k: round(v, 4) if isinstance(v, float) else v
                             for k, v in c.items()} for c in comparisons],
    }
    with open(os.path.join(output_dir, "recommendation.json"), "w") as f:
        json.dump(recommendation, f, indent=2)

    print(f"\n{'='*60}")
    print("FEATURE SELECTION SUMMARY")
    print(f"{'='*60}")
    print(f"  Best feature set: {best['name']} ({best['n_features']} features)")
    print(f"  Accuracy: {best['accuracy']:.4f}")
    print(f"  ROC-AUC:  {best['roc_auc']:.4f}")
    print(f"  Removed {len(feature_sets['removed'])} features (high null / redundant)")
    print(f"\n  Results saved to: {output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
