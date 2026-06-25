import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split, ShuffleSplit, LeaveOneOut, cross_val_predict
from sklearn.metrics import accuracy_score, mean_absolute_error, precision_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier
)

from utils_MCLR import load_and_preprocess_data


def plot_confusion_matrices(y_true, dict_preds, title_suffix="", save=False):
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

    if save:
        os.makedirs("outputs/model_results/benchmark", exist_ok=True)
        plt.savefig(f"outputs/model_results/benchmark/confusion_matrix_{title_suffix}.png")
    else:
        plt.show()


def get_metrics(y_true, y_pred):
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred)
    }
    precision = precision_score(y_true, y_pred, average=None, labels=[1, 2, 3, 4, 5], zero_division=0)
    for i, p in enumerate(precision, start=1):
        metrics[f'Precision_Class_{i}'] = p
    return metrics


def _split_data(X, y, train_idx, test_idx):
    if hasattr(X, "iloc"):
        return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def run_train_test(models, X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    results = {}
    dict_preds_train = {}
    dict_preds_test = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        dict_preds_train[name] = train_pred
        dict_preds_test[name] = test_pred

        train_metrics = get_metrics(y_train, train_pred)
        test_metrics = get_metrics(y_test, test_pred)

        results[name] = {}
        for k, v in train_metrics.items():
            results[name][f'Train {k}'] = v
        for k, v in test_metrics.items():
            results[name][f'Test {k}'] = v

    return pd.DataFrame(results).T, dict_preds_train, y_train, dict_preds_test, y_test


def run_shufflesplit(models, X, y, n_splits=5, test_size=0.2, random_state=42):
    cv = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    results = {}
    dict_preds_train = {}
    dict_preds_test = {}

    splits = list(cv.split(X, y))
    y_train_concat = []
    y_test_concat = []
    for train_idx, test_idx in splits:
        _, _, y_train, y_test = _split_data(X, y, train_idx, test_idx)
        y_train_concat.extend(y_train)
        y_test_concat.extend(y_test)

    y_train_concat = np.array(y_train_concat)
    y_test_concat = np.array(y_test_concat)

    for name, model in models.items():
        fold_metrics = []
        all_train_preds = []
        all_test_preds = []

        for train_idx, test_idx in splits:
            X_train, X_test, y_train, y_test = _split_data(X, y, train_idx, test_idx)
            model.fit(X_train, y_train)

            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            all_train_preds.extend(train_pred)
            all_test_preds.extend(test_pred)
            fold_metrics.append(get_metrics(y_test, test_pred))

        df_folds = pd.DataFrame(fold_metrics)
        results[name] = df_folds.mean().to_dict()
        dict_preds_train[name] = np.array(all_train_preds)
        dict_preds_test[name] = np.array(all_test_preds)

    return pd.DataFrame(results).T, dict_preds_train, y_train_concat, dict_preds_test, y_test_concat


def run_loo(models, X, y):
    cv = LeaveOneOut()
    results = {}
    dict_preds_train = {}
    dict_preds_test = {}

    splits = list(cv.split(X, y))
    y_train_concat = []
    for train_idx, _ in splits:
        if hasattr(y, "iloc"):
            y_train_concat.extend(y.iloc[train_idx])
        else:
            y_train_concat.extend(y[train_idx])
    y_train_concat = np.array(y_train_concat)

    for name, model in models.items():
        all_test_preds = np.zeros(len(y))
        all_train_preds = []

        for train_idx, test_idx in splits:
            X_train, X_test, y_train, y_test = _split_data(X, y, train_idx, test_idx)
            model.fit(X_train, y_train)

            test_pred = model.predict(X_test)
            train_pred = model.predict(X_train)

            all_test_preds[test_idx] = test_pred[0]
            all_train_preds.extend(train_pred)

        results[name] = get_metrics(y, all_test_preds)
        dict_preds_train[name] = np.array(all_train_preds)
        dict_preds_test[name] = all_test_preds

    return pd.DataFrame(results).T, dict_preds_train, y_train_concat, dict_preds_test, y

def load_and_prepare_data(base_csv, personnes_csv, photos_csv):
    df, _, _ = load_and_preprocess_data(base_csv, personnes_csv, photos_csv)
    liste_cat = ["nbr_lane", "speed", "slope", "type", "green"]
    X = df[liste_cat].copy()

    # Encodage en variables numériques lues par tous les modèles
    X = pd.get_dummies(X, columns=liste_cat, drop_first=True)

    y = df["note"].astype(int)
    return X, y

def get_models():
    return {
        "Classification Tree": DecisionTreeClassifier(random_state=42),
        "Gradient Boost": GradientBoostingClassifier(random_state=42),
        "HistGradient Boost": HistGradientBoostingClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Ada Boost": AdaBoostClassifier(random_state=42)
    }