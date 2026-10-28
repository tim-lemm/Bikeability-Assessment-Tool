import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import randint, uniform

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    ConfusionMatrixDisplay
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier
)
from sklearn.dummy import DummyClassifier
from utils_survey_analysis import import_and_merge_data_base

from config import *

# ==========================================
# 1. DATA PREPARATION
# ==========================================

def load_and_preprocess_data(base_path, personnes_path, photos_path, drop="first", people=False):
    df_all = import_and_merge_data_base(base_path, personnes_path, photos_path)
    df_all = df_all[df_all['job']==6]
    df_all = df_all[df_all['postal_code'].between(69000, 70000, inclusive='neither')]
    df = df_all.copy()
    df['id_personne'] = df['id_personne'].astype('category').cat.codes
    df['id_photo'] = df['id_photo'].astype('category').cat.codes
    if people:
        cat_cols = ['age', 'gender', 'electric_bike', 'speed', 'slope', 'green', 'type',
                'bike_use_frequency', 'bike_ownership', 'nbr_lane']
    else:
        cat_cols = ['speed', 'slope', 'green', 'type', 'nbr_lane']

    encoder_cat = OneHotEncoder(dtype=np.int64, drop=drop)
    enc_output = encoder_cat.fit_transform(df[cat_cols]).toarray()
    encoded_cols = encoder_cat.get_feature_names_out(cat_cols)

    df_enc = pd.DataFrame(enc_output, columns=encoded_cols, index=df.index)
    df_enc['id_personne'] = df['id_personne']
    df_enc['id_photo'] = df['id_photo']
    df_enc['note'] = df['note'].values - 1

    return df_enc, encoded_cols


def load_and_prepare_data(base_csv, personnes_csv, photos_csv, people = False):
    df, encoded_cols= load_and_preprocess_data(base_csv, personnes_csv, photos_csv, drop=None, people = people)

    X = df[encoded_cols].copy()
    y = df["note"].astype(int)

    return X, y

# ==========================================
# MODEL CONSTRUCTION
# ==========================================

def get_models_ML():
    return {
        "DummyClassifier_0": {
            "model": DummyClassifier(random_state=RANDOM_STATE, strategy="constant", constant=0),
            "params": {}
        },
        "DummyClassifier_1": {
            "model": DummyClassifier(random_state=RANDOM_STATE, strategy="constant", constant=1),
            "params": {}
        },
        "DummyClassifier_2": {
            "model": DummyClassifier(random_state=RANDOM_STATE, strategy="constant", constant=2),
            "params": {}
        },
        "DummyClassifier_3": {
            "model": DummyClassifier(random_state=RANDOM_STATE, strategy="constant", constant=3),
            "params": {}
        },
        "DummyClassifier_4": {
            "model": DummyClassifier(random_state=RANDOM_STATE, strategy="constant", constant=4),
            "params": {}
        },
        "DummyClassifier_strat": {
            "model": DummyClassifier(random_state=RANDOM_STATE, strategy="stratified"),
            "params": {}
        },
        "DummyClassifier_uniform": {
            "model": DummyClassifier(random_state=RANDOM_STATE, strategy="uniform"),
            "params": {}
        }

        # "Classification Tree": {
        #     "model": DecisionTreeClassifier(random_state=RANDOM_STATE),
        #     "params": {
        #         "max_depth": randint(1, 20),
        #         "min_samples_split": randint(2, 10),
        #         "min_samples_leaf": randint(1, 10),
        #     },
        # },
        # "Gradient Boost": {
        #     "model": GradientBoostingClassifier(random_state=RANDOM_STATE),
        #     "params": {
        #         "n_estimators": randint(50, 300),
        #         "learning_rate": uniform(0.01, 0.2),
        #         "max_depth": randint(1, 20),
        #         "subsample": uniform(0.6, 0.4),
        #     },
        # },
        # "HistGradient Boost": {
        #     "model": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        #     "params": {
        #         "learning_rate": uniform(0.01, 0.2),
        #         "max_iter": randint(50, 300),
        #         "max_depth": randint(1, 20),
        #         "l2_regularization": uniform(0.0, 1.0),
        #     },
        # },
        # "Random Forest": {
        #     "model": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        #     "params": {
        #         "n_estimators": randint(50, 300),
        #         "max_depth": randint(1, 20),
        #         "min_samples_split": randint(2, 10),
        #         "min_samples_leaf": randint(1, 10),
        #     },
        # },
        # "Ada Boost": {
        #     "model": AdaBoostClassifier(random_state=RANDOM_STATE),
        #     "params": {
        #         "n_estimators": randint(50, 300),
        #         "learning_rate": uniform(0.01, 1.0),
        #     },
        # },
        # "ExtraTrees": {
        #     "model": ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        #     "params": {
        #         "n_estimators": randint(50, 300),
        #         "max_depth": randint(1, 20),
        #         "min_samples_split": randint(2, 10),
        #         "min_samples_leaf": randint(1, 10),
        #     },
        # },
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