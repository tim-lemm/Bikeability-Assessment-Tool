import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import mode, randint, uniform

from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import (
    train_test_split, LeaveOneOut, StratifiedKFold,
    StratifiedShuffleSplit, GridSearchCV, ShuffleSplit
)
from sklearn.metrics import (
    accuracy_score, mean_absolute_error, precision_score,
    confusion_matrix, balanced_accuracy_score, recall_score,
    f1_score, cohen_kappa_score, classification_report, ConfusionMatrixDisplay
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier
)

from utils_survey_analysis import import_and_merge_data_base

from config import *

# ==========================================
# 1. DATA PREPARATION
# ==========================================

def load_and_preprocess_data(base_path, personnes_path, photos_path, drop="first"):
    df_all = import_and_merge_data_base(base_path, personnes_path, photos_path)
    df_all = df_all[df_all['job']==6]
    df_all = df_all[df_all['postal_code'].between(69000, 70000, inclusive='neither')]
    df = df_all.copy()
    df['id_personne'] = df['id_personne'].astype('category').cat.codes
    df['id_photo'] = df['id_photo'].astype('category').cat.codes
    # cat_cols = ['age', 'gender', 'job', 'electric_bike', 'speed', 'slope', 'green', 'type',
    #             'bike_use_frequency', 'bike_ownership', 'nbr_lane']
    cat_cols = ['age', 'gender', 'electric_bike', 'speed', 'slope', 'green', 'type',
                       'bike_use_frequency', 'bike_ownership', 'nbr_lane']
    encoder_cat = OneHotEncoder(dtype=np.int64, drop=drop)
    enc_output = encoder_cat.fit_transform(df[cat_cols]).toarray()
    encoded_cols = encoder_cat.get_feature_names_out(cat_cols)
    df_enc = pd.DataFrame(enc_output, columns=encoded_cols, index=df.index)
    df_enc['id_personne'] = df['id_personne']
    df_enc['id_photo'] = df['id_photo']

    df_enc['note'] = df['note'].values - 1

    cat_counts = {col: len(cats) for col, cats in zip(cat_cols, encoder_cat.categories_)}

    dims = {
        "n_personne": df_enc["id_personne"].nunique(),
        "n_genders": cat_counts["gender"],
        "n_bike_use_frequency": cat_counts["bike_use_frequency"],
        "n_ages": cat_counts["age"],
        # "n_jobs": cat_counts["job"],
        "n_bike_ownership": cat_counts["bike_ownership"],
        "n_electric_bikes": cat_counts["electric_bike"],
        "n_photos": df_enc["id_photo"].nunique(),
        "n_types": cat_counts["type"],
        "n_speeds": cat_counts["speed"],
        "n_nbr_lanes": cat_counts["nbr_lane"],
        "n_slopes": cat_counts["slope"],
        "n_greens": cat_counts["green"],
        "n_notes": 5
    }

    coords = {
        "categories_separation": list(encoder_cat.categories_[2]),
        "categories_gender": list(encoder_cat.categories_[3])
    }

    return df_enc, dims, coords


def load_and_prepare_data(base_csv, personnes_csv, photos_csv):
    df, _, _ = load_and_preprocess_data(base_csv, personnes_csv, photos_csv, drop=None)

    liste_cat = [f"nbr_lane_{i}" for i in range(4)] + [f"speed_{i}" for i in range(4)] + \
                [f"slope_{i}" for i in range(3)] + [f"green_{i}" for i in range(3)] + \
                [f"type_{i}" for i in range(4)]

    X = df[liste_cat].copy()
    y = df["note"].astype(int)

    df, dims, coords = load_and_preprocess_data(base_csv, personnes_csv, photos_csv, drop="first")

    return X, y, df, dims, coords

# ==========================================
# MODEL CONSTRUCTION
# ==========================================

def get_models_ML():
    return {
        "Classification Tree": {
            "model": DecisionTreeClassifier(random_state=RANDOM_STATE),
            "params": {
                "max_depth": randint(1, 20),
                "min_samples_split": randint(2, 10),
                "min_samples_leaf": randint(1, 10),
            },
        },
        "Gradient Boost": {
            "model": GradientBoostingClassifier(random_state=RANDOM_STATE),
            "params": {
                "n_estimators": randint(50, 300),
                "learning_rate": uniform(0.01, 0.2),
                "max_depth": randint(1, 20),
                "subsample": uniform(0.6, 0.4),
            },
        },
        "HistGradient Boost": {
            "model": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
            "params": {
                "learning_rate": uniform(0.01, 0.2),
                "max_iter": randint(50, 300),
                "max_depth": randint(1, 20),
                "l2_regularization": uniform(0.0, 1.0),
            },
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            "params": {
                "n_estimators": randint(50, 300),
                "max_depth": randint(1, 20),
                "min_samples_split": randint(2, 10),
                "min_samples_leaf": randint(1, 10),
            },
        },
        "Ada Boost": {
            "model": AdaBoostClassifier(random_state=RANDOM_STATE),
            "params": {
                "n_estimators": randint(50, 300),
                "learning_rate": uniform(0.01, 1.0),
            },
        },
        "ExtraTrees": {
            "model": ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            "params": {
                "n_estimators": randint(50, 300),
                "max_depth": randint(1, 20),
                "min_samples_split": randint(2, 10),
                "min_samples_leaf": randint(1, 10),
            },
        },
    }

# ==========================================
# PLOTS AND VISUALIZATIONS
# ==========================================

def plot_all_confusion_matrices(models_dict, X_test, y_test, output_dir="", save=False):
  n_models = len(models_dict)
  cols = 3
  rows = (n_models + cols - 1) // cols

  fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
  axes = axes.flatten()

  for i, (name, model) in enumerate(models_dict.items()):
    ax = axes[i]
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, ax=ax, cmap="Greens", colorbar=False
    )
    ax.set_title(name)

  for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

  plt.tight_layout()
  if save:
    plot_path = os.path.join(output_dir, "confusion_matrices.png")
    plt.savefig(plot_path)
    print(f" Matrices de confusion sauvegardées dans : {plot_path}")
  plt.show()