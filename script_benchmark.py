import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed

from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.base import clone
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier
)
from sklearn.metrics import mean_absolute_error, accuracy_score, precision_score, confusion_matrix

from utils_MCLR import load_and_preprocess_data


def load_and_prepare_data(base_csv, personnes_csv, photos_csv):
    df, _, _ = load_and_preprocess_data(base_csv, personnes_csv, photos_csv)
    liste_cat = ["nbr_lane", "speed", "slope", "type", "green"]
    X = df[liste_cat].copy()

    for cat in liste_cat:
        X[cat] = X[cat].astype("category")

    y = df["note"]
    return X, y


def get_models():
    return {
        "Tree Regression": DecisionTreeRegressor(random_state=42),
        "Classification Tree": DecisionTreeClassifier(random_state=42),
        "Gradient Boost": GradientBoostingClassifier(random_state=42),
        "HistGradient Boost": HistGradientBoostingClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Ada Boost": AdaBoostClassifier(random_state=42)
    }


def compute_metrics(y_true, y_pred_discrete, y_pred_raw):
    mae = mean_absolute_error(y_true, y_pred_raw)
    acc = accuracy_score(y_true, y_pred_discrete)
    precisions = precision_score(y_true, y_pred_discrete, labels=[1, 2, 3, 4, 5], average=None, zero_division=0)

    return {
        "MAE": round(mae, 3),
        "Accuracy": round(acc, 3),
        "Prec_Note_1": round(precisions[0], 3),
        "Prec_Note_2": round(precisions[1], 3),
        "Prec_Note_3": round(precisions[2], 3),
        "Prec_Note_4": round(precisions[3], 3),
        "Prec_Note_5": round(precisions[4], 3)
    }


def plot_confusion_matrices(y_true, dict_preds, title_suffix=""):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.ravel()
    labels = [1, 2, 3, 4, 5]

    for idx, (name, y_pred) in enumerate(dict_preds.items()):
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=labels, yticklabels=labels, ax=axes[idx],
                    cbar=False)
        axes[idx].set_title(name, fontsize=12, fontweight='bold')
        axes[idx].set_ylabel("Notes Réelles")
        axes[idx].set_xlabel("Notes Prédites")

    plt.suptitle(f"Matrices de Confusion - {title_suffix}", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()


def benchmark_train_test(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = get_models()
    results = []
    all_preds = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_raw = model.predict(X_test)
        y_pred_discrete = np.clip(np.round(y_pred_raw), 1, 5).astype(int)

        all_preds[name] = y_pred_discrete
        metrics = compute_metrics(y_test, y_pred_discrete, y_pred_raw)
        results.append({"Modèle": name, **metrics})

    return pd.DataFrame(results), y_test, all_preds


def _run_single_fold(train_idx, test_idx, X, y, model):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model_fold = clone(model)
    model_fold.fit(X_train, y_train)
    return y_test.values[0], model_fold.predict(X_test)[0]


def benchmark_loocv(X, y):
    models = get_models()
    loo = LeaveOneOut()
    results = []
    all_preds = {}
    y_true_global = None

    for name, model in models.items():
        fold_results = Parallel(n_jobs=-1)(
            delayed(_run_single_fold)(train_idx, test_idx, X, y, model)
            for train_idx, test_idx in loo.split(X)
        )

        y_true_all, y_pred_all = zip(*fold_results)
        y_true_all = np.array(y_true_all)
        y_pred_raw = np.array(y_pred_all)
        y_pred_discrete = np.clip(np.round(y_pred_raw), 1, 5).astype(int)

        if y_true_global is None:
            y_true_global = y_true_all

        all_preds[name] = y_pred_discrete
        metrics = compute_metrics(y_true_all, y_pred_discrete, y_pred_raw)
        results.append({"Modèle": name, **metrics})

    return pd.DataFrame(results), y_true_global, all_preds


if __name__ == "__main__":
    base_csv = "data/survey/data-base.csv"
    personnes_csv = "data/survey/data-personnes.csv"
    photos_csv = "data/survey/data-photos.csv"

    X, y = load_and_prepare_data(base_csv, personnes_csv, photos_csv)

    print("=== Benchmark : Train / Test ===")
    df_results_tt, y_test_tt, preds_tt = benchmark_train_test(X, y)
    print(df_results_tt.to_string(index=False))
    plot_confusion_matrices(y_test_tt, preds_tt, title_suffix="Train / Test Split")

    # print("\n=== Benchmark : LOOCV ===")
    # df_results_loocv, y_true_loo, preds_loo = benchmark_loocv(X, y)
    # print(df_results_loocv.to_string(index=False))
    # plot_confusion_matrices(y_true_loo, preds_loo, title_suffix="LOOCV")