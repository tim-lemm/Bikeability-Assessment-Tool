import os
import numpy as np
import pandas as pd
import pymc as pm
import pymc.distributions.transforms as tr
import arviz as az
import xarray as xr
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import mode

from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import (
    train_test_split, LeaveOneOut, StratifiedKFold,
    StratifiedShuffleSplit, GridSearchCV, ShuffleSplit
)
from sklearn.metrics import (
    accuracy_score, mean_absolute_error, precision_score,
    confusion_matrix, balanced_accuracy_score, recall_score,
    f1_score, cohen_kappa_score, classification_report
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


# ==========================================
# 1. DATA PREPARATION
# ==========================================

def load_and_preprocess_data(base_path, personnes_path, photos_path, drop="first"):
    df = import_and_merge_data_base(base_path, personnes_path, photos_path)
    df['id_personne'] = df['id_personne'].astype('category').cat.codes
    df['id_photo'] = df['id_photo'].astype('category').cat.codes
    cat_cols = ['age', 'gender', 'job', 'electric_bike', 'speed', 'slope', 'green', 'type',
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
        "n_jobs": cat_counts["job"],
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


def _split_data(X, y, train_idx, test_idx):
    if hasattr(X, "iloc"):
        return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# ==========================================
# 2. MODEL CONSTRUCTION
# ==========================================

def get_models_ML():
    return {
        "Classification Tree": DecisionTreeClassifier(random_state=42),
        "Gradient Boost": GradientBoostingClassifier(random_state=42, learning_rate=0.2,
                                                     max_depth=5, min_samples_split=2,
                                                     min_samples_leaf=1, n_estimators=500,
                                                     subsample=0.8),
        "HistGradient Boost": HistGradientBoostingClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Ada Boost": AdaBoostClassifier(random_state=42),
        "ExtraTrees": ExtraTreesClassifier(random_state=42)
    }


def get_models_gb():
    return {
        "Gradient Boost 0": GradientBoostingClassifier(random_state=42),
        "Gradient Boost 1": GradientBoostingClassifier(random_state=42, max_depth=1),
        "Gradient Boost 2": GradientBoostingClassifier(random_state=42, max_depth=2),
        "Gradient Boost 3": GradientBoostingClassifier(random_state=42, max_depth=3),
        "Gradient Boost 4": GradientBoostingClassifier(random_state=42, max_depth=4),
        "Gradient Boost 5": GradientBoostingClassifier(random_state=42, max_depth=5)
    }


def build_model_1(encoded_df, dims):
    # Extract NumPy matrices
    X_age_np = encoded_df[[col for col in encoded_df.columns if col.startswith('age_')]].values
    X_gender_np = encoded_df[[col for col in encoded_df.columns if col.startswith('gender_')]].values
    X_job_np = encoded_df[[col for col in encoded_df.columns if col.startswith('job_')]].values
    X_electric_bike_np = encoded_df[[col for col in encoded_df.columns if col.startswith('electric_bike_')]].values
    X_bike_use_frequency_np = encoded_df[
        [col for col in encoded_df.columns if col.startswith('bike_use_frequency_')]].values
    X_bike_ownership_np = encoded_df[[col for col in encoded_df.columns if col.startswith('bike_ownership_')]].values

    X_nbr_lanes_np = encoded_df[[col for col in encoded_df.columns if col.startswith('nbr_lane_')]].values
    X_type_np = encoded_df[[col for col in encoded_df.columns if col.startswith('type_')]].values
    X_slope_np = encoded_df[[col for col in encoded_df.columns if col.startswith('slope_')]].values
    X_speed_np = encoded_df[[col for col in encoded_df.columns if col.startswith('speed_')]].values
    X_green_np = encoded_df[[col for col in encoded_df.columns if col.startswith('green_')]].values

    id_personne_np = encoded_df['id_personne'].values
    id_photo_np = encoded_df['id_photo'].values
    note_np = encoded_df['note'].values

    with pm.Model() as model:
        X_age = pm.Data("X_age", X_age_np)
        X_gender = pm.Data("X_gender", X_gender_np)
        X_job = pm.Data("X_job", X_job_np)
        X_electric_bike = pm.Data("X_electric_bike", X_electric_bike_np)
        X_bike_use_frequency = pm.Data("X_bike_use_frequency", X_bike_use_frequency_np)
        X_bike_ownership = pm.Data("X_bike_ownership", X_bike_ownership_np)

        X_nbr_lanes = pm.Data("X_nbr_lanes", X_nbr_lanes_np)
        X_type = pm.Data("X_type", X_type_np)
        X_slope = pm.Data("X_slope", X_slope_np)
        X_speed = pm.Data("X_speed", X_speed_np)
        X_green = pm.Data("X_green", X_green_np)

        id_personne = pm.Data("id_personne", id_personne_np)
        id_photo = pm.Data("id_photo", id_photo_np)
        y_obs_container = pm.Data("y_obs_data", note_np)

        # Priors (Unchanged)
        cutpoints = pm.Normal('cutpoints', mu=np.linspace(-2, 2, dims["n_notes"] - 1), sigma=1,
                              transform=pm.distributions.transforms.ordered, shape=dims["n_notes"] - 1)

        beta_age = pm.Normal("beta_age", mu=0, sigma=1, shape=dims["n_ages"] - 1)
        beta_gender = pm.Normal("beta_gender", mu=0, sigma=1, shape=dims["n_genders"] - 1)
        beta_job = pm.Normal("beta_job", mu=0, sigma=1, shape=dims["n_jobs"] - 1)
        beta_electric_bike = pm.Normal("beta_electric_bike", mu=0, sigma=1, shape=dims["n_electric_bikes"] - 1)
        beta_bike_use_frequency = pm.Normal("beta_bike_use_frequency", mu=0, sigma=1,
                                            shape=dims["n_bike_use_frequency"] - 1)
        beta_bike_ownership = pm.Normal("beta_bike_ownership", mu=0, sigma=1, shape=dims["n_bike_ownership"] - 1)

        beta_nbr_lanes = pm.Normal("beta_nbr_lanes", mu=0, sigma=1, shape=dims["n_nbr_lanes"] - 1)
        beta_type = pm.Normal("beta_type", mu=0, sigma=1, shape=dims["n_types"] - 1)
        beta_slope = pm.Normal("beta_slope", mu=0, sigma=1, shape=dims["n_slopes"] - 1)
        beta_speed = pm.Normal("beta_speed", mu=0, sigma=1, shape=dims["n_speeds"] - 1)
        beta_green = pm.Normal("beta_green", mu=0, sigma=1, shape=dims["n_greens"] - 1)

        # Random effects (Fixed size based on training)
        sigma_personne = pm.HalfNormal("sigma_personne", sigma=1)
        u_personne = pm.Normal("u_personne", mu=0, sigma=sigma_personne, shape=dims["n_personne"])
        sigma_photo = pm.HalfNormal("sigma_photo", sigma=1)
        v_photo = pm.Normal("v_photo", mu=0, sigma=sigma_photo, shape=dims["n_photos"])

        id_personne_safe = pm.math.switch(id_personne < dims["n_personne"], id_personne, 0)
        id_photo_safe = pm.math.switch(id_photo < dims["n_photos"], id_photo, 0)

        eff_personne = u_personne[id_personne_safe]
        eff_photo = v_photo[id_photo_safe]

        # Linear combination (using MutableData)
        mu = (
                eff_personne + pm.math.dot(X_age, beta_age) + pm.math.dot(X_gender, beta_gender) +
                pm.math.dot(X_job, beta_job) + pm.math.dot(X_electric_bike, beta_electric_bike) +
                pm.math.dot(X_bike_use_frequency, beta_bike_use_frequency) + pm.math.dot(X_bike_ownership,
                                                                                         beta_bike_ownership) +
                eff_photo + pm.math.dot(X_nbr_lanes, beta_nbr_lanes) + pm.math.dot(X_type, beta_type) +
                pm.math.dot(X_slope, beta_slope) + pm.math.dot(X_speed, beta_speed) + pm.math.dot(X_green, beta_green)
        )

        pm.OrderedLogistic("y_obs", eta=mu, cutpoints=cutpoints, observed=y_obs_container)

    return model


def build_model_2(encoded_df, dims):
    X_nbr_lanes_np = encoded_df[[col for col in encoded_df.columns if col.startswith('nbr_lane_')]].values
    X_type_np = encoded_df[[col for col in encoded_df.columns if col.startswith('type_')]].values
    X_slope_np = encoded_df[[col for col in encoded_df.columns if col.startswith('slope_')]].values
    X_speed_np = encoded_df[[col for col in encoded_df.columns if col.startswith('speed_')]].values
    X_green_np = encoded_df[[col for col in encoded_df.columns if col.startswith('green_')]].values

    id_photo_np = encoded_df['id_photo'].values
    note_np = encoded_df['note'].values

    with pm.Model() as model:
        X_nbr_lanes = pm.Data("X_nbr_lanes", X_nbr_lanes_np)
        X_type = pm.Data("X_type", X_type_np)
        X_slope = pm.Data("X_slope", X_slope_np)
        X_speed = pm.Data("X_speed", X_speed_np)
        X_green = pm.Data("X_green", X_green_np)
        id_photo = pm.Data("id_photo", id_photo_np)
        y_obs_container = pm.Data("y_obs_data", note_np)

        cutpoints = pm.Normal('cutpoints', mu=np.linspace(-2, 2, dims["n_notes"] - 1), sigma=1,
                              transform=pm.distributions.transforms.ordered, shape=dims["n_notes"] - 1)

        beta_nbr_lanes = pm.Normal("beta_nbr_lanes", mu=0, sigma=1, shape=dims["n_nbr_lanes"] - 1)
        beta_type = pm.Normal("beta_type", mu=0, sigma=1, shape=dims["n_types"] - 1)
        beta_slope = pm.Normal("beta_slope", mu=0, sigma=1, shape=dims["n_slopes"] - 1)
        beta_speed = pm.Normal("beta_speed", mu=0, sigma=1, shape=dims["n_speeds"] - 1)
        beta_green = pm.Normal("beta_green", mu=0, sigma=1, shape=dims["n_greens"] - 1)

        sigma_photo = pm.HalfNormal("sigma_photo", sigma=1)
        v_photo = pm.Normal("v_photo", mu=0, sigma=sigma_photo, shape=dims["n_photos"])
        id_photo_safe = pm.math.switch(id_photo < dims["n_photos"], id_photo, 0)
        eff_photo = v_photo[id_photo_safe]

        mu = (
                eff_photo + pm.math.dot(X_nbr_lanes, beta_nbr_lanes) + pm.math.dot(X_type, beta_type) +
                pm.math.dot(X_slope, beta_slope) + pm.math.dot(X_speed, beta_speed) + pm.math.dot(X_green, beta_green)
        )

        pm.OrderedLogistic("y_obs", eta=mu, cutpoints=cutpoints, observed=y_obs_container)

    return model


def get_models_MCLR(df, dims):
    return {
        "MCLR Model 1": [build_model_1(df, dims),
                         ["beta_age", "beta_gender", "beta_job", "beta_electric_bike", "beta_bike_use_frequency",
                          "beta_bike_ownership", "beta_nbr_lanes", "beta_type", "beta_slope", "beta_speed",
                          "beta_green"]],
        "MCLR Model 2": [build_model_2(df, dims),
                         ["beta_nbr_lanes", "beta_type", "beta_slope", "beta_speed", "beta_green"]]
    }


# ==========================================
# 3. EVALUATION AND METRICS
# ==========================================

def get_mclr_probabilities(idata, y_true=None, var_name="y_obs", num_classes=5):
    """
    Calcule les probabilités de prédiction pour chaque classe, inclut la note réelle
    et la note prédite pour chaque observation.
    """
    # 1. Extraire les prédictions (shape: chains, draws, n_obs)
    y_pred = idata.posterior_predictive[var_name].values

    # 2. Aplatir les dimensions chains et draws (shape: n_samples, n_obs)
    y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])
    n_samples, n_obs = y_pred_flat.shape

    # 3. Calculer la proportion (probabilité) de chaque classe
    probs = np.zeros((n_obs, num_classes))
    for c in range(num_classes):
        probs[:, c] = np.sum(y_pred_flat == c, axis=0) / n_samples

    # 4. Formater la sortie dans un DataFrame Pandas
    df_probs = pd.DataFrame(
        probs,
        columns=[f"Prob_Note_{c + 1}" for c in range(num_classes)]
    )

    # 5. Ajouter la note prédite (la classe avec la probabilité max, +1 pour l'échelle 1-5)
    df_probs["Note_Predite"] = np.argmax(probs, axis=1) + 1

    # 6. Ajouter la note réelle si elle est fournie (+1 pour l'échelle 1-5)
    if y_true is not None:
        if hasattr(y_true, "values"):
            y_true_vals = y_true.values
        else:
            y_true_vals = np.array(y_true)

        df_probs["Note_Reelle"] = y_true_vals + 1

    return df_probs

def get_metrics_ML(y_true, y_pred):
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='weighted'),
        'F1': f1_score(y_true, y_pred, average='weighted'),
        'Cohen Kappa': cohen_kappa_score(y_true, y_pred)
    }
    precision = precision_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3, 4], zero_division=0)
    for i, p in enumerate(precision, start=0):
        metrics[f'Precision_Class_{i}'] = p
    return metrics


def get_metrics_MCLR(y_true, idata):
    var_name = "y_obs"
    y_pred_samples = idata.posterior_predictive[var_name].values
    y_pred_samples = y_pred_samples.reshape(-1, y_pred_samples.shape[-1]).T
    num_samples = y_pred_samples.shape[1]

    acc_list, bal_acc_list, mae_list, prec_list, rec_list, f1_list, kappa_list = [], [], [], [], [], [], []
    num_class = 5
    prec_class_lists = {i: [] for i in range(num_class)}

    for j in range(num_samples):
        yp = y_pred_samples[:, j]
        acc_list.append(accuracy_score(yp, y_true))
        bal_acc_list.append(balanced_accuracy_score(yp, y_true))
        mae_list.append(mean_absolute_error(yp, y_true))
        prec_list.append(precision_score(yp, y_true, average='weighted', zero_division=0))
        rec_list.append(recall_score(yp, y_true, average='weighted', zero_division=0))
        f1_list.append(f1_score(yp, y_true, average='weighted', zero_division=0))
        kappa_list.append(cohen_kappa_score(yp, y_true))

        p_class = precision_score(yp, y_true, average=None, labels=list(range(num_class)), zero_division=0)
        for i, p in enumerate(p_class):
            prec_class_lists[i].append(p)

    metrics = {
        "Accuracy": np.mean(acc_list),
        "Balanced Accuracy": np.mean(bal_acc_list),
        "MAE": np.mean(mae_list),
        "Precision": np.mean(prec_list),
        "Recall": np.mean(rec_list),
        "F1": np.mean(f1_list),
        "Cohen Kappa": np.mean(kappa_list)
    }

    for i in range(num_class):
        metrics[f'Precision_Class_{i}'] = np.mean(prec_list[i])

    return metrics


# ==========================================
# 4. EXECUTION (ML, MCLR, LOO, SHUFFLESPLIT)
# ==========================================

def run_train_test_ML(models, X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    results, dict_preds_train, dict_preds_test = {}, {}, {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        dict_preds_train[name] = train_pred
        dict_preds_test[name] = test_pred

        train_metrics = get_metrics_ML(y_train, train_pred)
        test_metrics = get_metrics_ML(y_test, test_pred)

        results[name] = {}
        for k, v in train_metrics.items():
            results[name][f'Train {k}'] = v
        for k, v in test_metrics.items():
            results[name][f'Test {k}'] = v

    return pd.DataFrame(results).T, dict_preds_train, y_train, dict_preds_test, y_test


def run_train_test_MCLR(df, dims, test_size=0.2, random_state=42):
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    df_train, df_test = pd.concat([X_train, y_train], axis=1), pd.concat([X_test, y_test], axis=1)

    results = {}
    train_predictions_for_plots = {}
    test_predictions_for_plots = {}
    test_probabilities_all = {}
    train_probabilities_all = {}

    def get_test_arrays(df_test):
        return {
            "X_age": df_test[[col for col in df_test.columns if col.startswith('age_')]].values,
            "X_gender": df_test[[col for col in df_test.columns if col.startswith('gender_')]].values,
            "X_job": df_test[[col for col in df_test.columns if col.startswith('job_')]].values,
            "X_electric_bike": df_test[[col for col in df_test.columns if col.startswith('electric_bike_')]].values,
            "X_bike_use_frequency": df_test[
                [col for col in df_test.columns if col.startswith('bike_use_frequency_')]].values,
            "X_bike_ownership": df_test[[col for col in df_test.columns if col.startswith('bike_ownership_')]].values,
            "X_nbr_lanes": df_test[[col for col in df_test.columns if col.startswith('nbr_lane_')]].values,
            "X_type": df_test[[col for col in df_test.columns if col.startswith('type_')]].values,
            "X_slope": df_test[[col for col in df_test.columns if col.startswith('slope_')]].values,
            "X_speed": df_test[[col for col in df_test.columns if col.startswith('speed_')]].values,
            "X_green": df_test[[col for col in df_test.columns if col.startswith('green_')]].values,
            "id_personne": df_test['id_personne'].values,
            "id_photo": df_test['id_photo'].values
        }

    X_test_arr = get_test_arrays(df_test)
    models = get_models_MCLR(df_train, dims)

    for name, model in models.items():
        model_actif = models[name][0]
        with model_actif:
            trace = pm.sample(draws=1000, tune=1000, return_inferencedata=True)
            idata_train = pm.sample_posterior_predictive(trace)

            data_dict = {
                "X_nbr_lanes": X_test_arr["X_nbr_lanes"], "X_type": X_test_arr["X_type"],
                "X_slope": X_test_arr["X_slope"], "X_speed": X_test_arr["X_speed"],
                "X_green": X_test_arr["X_green"], "id_photo": X_test_arr["id_photo"],
                "y_obs_data": y_test
            }
            if name == "MCLR Model 1":
                data_dict.update({
                    "X_age": X_test_arr["X_age"], "X_gender": X_test_arr["X_gender"],
                    "X_job": X_test_arr["X_job"], "X_electric_bike": X_test_arr["X_electric_bike"],
                    "X_bike_use_frequency": X_test_arr["X_bike_use_frequency"],
                    "X_bike_ownership": X_test_arr["X_bike_ownership"],
                    "id_personne": X_test_arr["id_personne"]
                })

            pm.set_data(data_dict)
            idata_test = pm.sample_posterior_predictive(trace)

            # --- CHANGEMENT ICI : On enregistre dans le dictionnaire avec [name] ---
            test_probabilities_all[name] = get_mclr_probabilities(
                idata_test,
                y_true=y_test,
                num_classes=dims["n_notes"]
            )
            train_probabilities_all[name] = get_mclr_probabilities(
                idata_train,
                y_true=y_train,
                num_classes=dims["n_notes"]
            )

        train_metrics = get_metrics_MCLR(y_train, idata_train)
        test_metrics = get_metrics_MCLR(y_test, idata_test)

        results[name] = {}
        for k, v in train_metrics.items(): results[name][f'Train {k}'] = v
        for k, v in test_metrics.items(): results[name][f'Test {k}'] = v

        y_pred_train_flat = idata_train["posterior_predictive"]["y_obs"].values.reshape(-1, idata_train[
            "posterior_predictive"]["y_obs"].values.shape[-1])
        train_predictions_for_plots[name] = mode(y_pred_train_flat, axis=0, keepdims=True).mode[0]

        y_pred_test_flat = idata_test["posterior_predictive"]["y_obs"].values.reshape(-1, idata_test[
            "posterior_predictive"]["y_obs"].values.shape[-1])
        test_predictions_for_plots[name] = mode(y_pred_test_flat, axis=0, keepdims=True).mode[0]


    return pd.DataFrame(
        results).T, test_probabilities_all, train_probabilities_all, train_predictions_for_plots, test_predictions_for_plots


def run_shufflesplit_ML(models, X, y, n_splits=5, test_size=0.2, random_state=42):
    cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    results, dict_preds_train, dict_preds_test = {}, {}, {}

    splits = list(cv.split(X, y))
    y_train_concat, y_test_concat = [], []
    for train_idx, test_idx in splits:
        _, _, y_train, y_test = _split_data(X, y, train_idx, test_idx)
        y_train_concat.extend(y_train)
        y_test_concat.extend(y_test)

    y_train_concat, y_test_concat = np.array(y_train_concat), np.array(y_test_concat)

    for name, model in models.items():
        fold_metrics, all_train_preds, all_test_preds = [], [], []

        for train_idx, test_idx in splits:
            X_train, X_test, y_train, y_test = _split_data(X, y, train_idx, test_idx)
            model.fit(X_train, y_train)

            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            all_train_preds.extend(train_pred)
            all_test_preds.extend(test_pred)

            combined_metrics = {}
            for k, v in get_metrics_ML(y_train, train_pred).items(): combined_metrics[f'Train {k}'] = v
            for k, v in get_metrics_ML(y_test, test_pred).items(): combined_metrics[f'Test {k}'] = v
            fold_metrics.append(combined_metrics)

        results[name] = pd.DataFrame(fold_metrics).mean().to_dict()
        dict_preds_train[name] = np.array(all_train_preds)
        dict_preds_test[name] = np.array(all_test_preds)

    return pd.DataFrame(results).T, dict_preds_train, y_train_concat, dict_preds_test, y_test_concat


def run_shufflesplit_MCLR(df, dims, n_splits=5, test_size=0.2, random_state=42):
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    models = get_models_MCLR(df, dims)
    cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    results, dict_preds_train, dict_preds_test = {}, {}, {}

    splits = list(cv.split(X, y))
    y_train_concat, y_test_concat = [], []

    for train_idx, test_idx in splits:
        y_train_concat.extend(y.iloc[train_idx].values)
        y_test_concat.extend(y.iloc[test_idx].values)

    y_train_concat, y_test_concat = np.array(y_train_concat), np.array(y_test_concat)

    def _get_pymc_data_dict(df_fold, model_actif, y_fold=None):
        data_dict = {
            "X_nbr_lanes": df_fold[[col for col in df_fold.columns if col.startswith('nbr_lane_')]].values,
            "X_type": df_fold[[col for col in df_fold.columns if col.startswith('type_')]].values,
            "X_slope": df_fold[[col for col in df_fold.columns if col.startswith('slope_')]].values,
            "X_speed": df_fold[[col for col in df_fold.columns if col.startswith('speed_')]].values,
            "X_green": df_fold[[col for col in df_fold.columns if col.startswith('green_')]].values,
            "id_photo": df_fold['id_photo'].values
        }
        if "X_age" in [v.name for v in model_actif.data_vars]:
            data_dict.update({
                "X_age": df_fold[[col for col in df_fold.columns if col.startswith('age_')]].values,
                "X_gender": df_fold[[col for col in df_fold.columns if col.startswith('gender_')]].values,
                "X_job": df_fold[[col for col in df_fold.columns if col.startswith('job_')]].values,
                "X_electric_bike": df_fold[[col for col in df_fold.columns if col.startswith('electric_bike_')]].values,
                "X_bike_use_frequency": df_fold[
                    [col for col in df_fold.columns if col.startswith('bike_use_frequency_')]].values,
                "X_bike_ownership": df_fold[
                    [col for col in df_fold.columns if col.startswith('bike_ownership_')]].values,
                "id_personne": df_fold['id_personne'].values
            })
        if y_fold is not None:
            data_dict["y_obs_data"] = y_fold.values
        return data_dict

    for name, model_info in models.items():
        model_actif = model_info[0]
        fold_metrics, all_train_preds, all_test_preds = [], [], []

        print(f"-> Cross-Validation training and evaluation of : {name}")
        for split_idx, (train_idx, test_idx) in enumerate(splits):
            print(f"   Fold {split_idx + 1}/{n_splits}...")
            df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]
            y_train, y_test = df_train.iloc[:, -1], df_test.iloc[:, -1]

            with model_actif:
                pm.set_data(_get_pymc_data_dict(df_train, model_actif, y_train))
                trace = pm.sample(draws=1000, tune=1000, return_inferencedata=True, progressbar=False)
                idata_train = pm.sample_posterior_predictive(trace, progressbar=False)

                y_pred_train_flat = idata_train["posterior_predictive"]["y_obs"].values.reshape(-1, idata_train[
                    "posterior_predictive"]["y_obs"].values.shape[-1])
                all_train_preds.extend(mode(y_pred_train_flat, axis=0, keepdims=True).mode[0])

                pm.set_data(_get_pymc_data_dict(df_test, model_actif, y_test))
                idata_test = pm.sample_posterior_predictive(trace, progressbar=False)

                y_pred_test_flat = idata_test["posterior_predictive"]["y_obs"].values.reshape(-1, idata_test[
                    "posterior_predictive"]["y_obs"].values.shape[-1])
                all_test_preds.extend(mode(y_pred_test_flat, axis=0, keepdims=True).mode[0])

            combined_metrics = {}
            for k, v in get_metrics_MCLR(y_train, idata_train).items(): combined_metrics[f'Train {k}'] = v
            for k, v in get_metrics_MCLR(y_test, idata_test).items(): combined_metrics[f'Test {k}'] = v
            fold_metrics.append(combined_metrics)

        results[name] = pd.DataFrame(fold_metrics).mean().to_dict()
        dict_preds_train[name] = np.array(all_train_preds)
        dict_preds_test[name] = np.array(all_test_preds)

    return pd.DataFrame(results).T, dict_preds_train, y_train_concat, dict_preds_test, y_test_concat


def run_loo(models, X, y):
    cv = LeaveOneOut()
    results, dict_preds_train, dict_preds_test = {}, {}, {}

    splits = list(cv.split(X, y))
    y_train_concat = []
    for train_idx, _ in splits:
        y_train_concat.extend(y.iloc[train_idx] if hasattr(y, "iloc") else y[train_idx])
    y_train_concat = np.array(y_train_concat)

    for name, model in models.items():
        all_test_preds, all_train_preds = np.zeros(len(y)), []

        for train_idx, test_idx in splits:
            X_train, X_test, y_train, y_test = _split_data(X, y, train_idx, test_idx)
            model.fit(X_train, y_train)
            all_test_preds[test_idx] = model.predict(X_test)[0]
            all_train_preds.extend(model.predict(X_train))

        results[name] = get_metrics_ML(y, all_test_preds)
        dict_preds_train[name] = np.array(all_train_preds)
        dict_preds_test[name] = all_test_preds

    return pd.DataFrame(results).T, dict_preds_train, y_train_concat, dict_preds_test, y


def optimize_model_hp(X, y, base_model, param_grid, scoring_metric="balanced_accuracy"):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("Starting hyperparameter search (GridSearchCV)...")

    grid_search = GridSearchCV(
        estimator=base_model, param_grid=param_grid, scoring=scoring_metric,
        cv=cv, n_jobs=-1, verbose=3, return_train_score=True
    )
    grid_search.fit(X, y)
    results_df = pd.DataFrame(grid_search.cv_results_)

    cols_to_keep = ['params', 'mean_train_score', 'mean_test_score']
    print(results_df[cols_to_keep].sort_values(by='mean_test_score', ascending=False))

    print("\n--- Optimization Results ---")
    print(f"Best score ({scoring_metric}) : {grid_search.best_score_}")
    print("Best hyperparameters found :", grid_search.best_params_)

    best_index = grid_search.best_index_
    print(f"\nTrain Score of the best model : {results_df.loc[best_index, 'mean_train_score']}")
    print(f"Test Score of the best model  : {results_df.loc[best_index, 'mean_test_score']}")

    return grid_search.best_estimator_, grid_search.best_params_


def run_benchmark(df, dims, coords, model_factories, n_splits=5, draws=1000, tune=1000):
    print(f"\n STARTING BENCHMARK (Train-Test Split 80/20 repeated {n_splits} times)...")
    rs = ShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)
    raw_results = {name: [] for name in model_factories.keys()}

    for fold, (train_idx, test_idx) in enumerate(rs.split(df)):
        print(f"\n --- FOLD {fold + 1} / {n_splits} ---")
        df_train, df_test = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()

        for model_name, model_builder in model_factories.items():
            print(f" Training [{model_name}]...")

            # 1. Fit on the training set
            train_model = model_builder(df_train, dims, coords)
            with train_model:
                idata_train = pm.sample(draws=draws, tune=tune, return_inferencedata=True, progressbar=False,
                                        random_seed=42)

            print(f"Prediction of [{model_name}] on the test set...")
            posterior_ds = idata_train["posterior"].dataset.drop_vars("y_obs_probs", errors="ignore")
            idata_posterior_only = xr.DataTree()
            idata_posterior_only["posterior"] = posterior_ds

            # 2. Model reconstruction with test data
            test_model = model_builder(df_test, dims, coords)
            with test_model:
                ppc = pm.sample_posterior_predictive(idata_posterior_only, progressbar=False, random_seed=42)

            # 3. Extraction and reduction of predictions (Median of draws)
            ppc_samples = ppc.posterior_predictive["y_obs"].values.reshape(-1, ppc.posterior_predictive[
                "y_obs"].values.shape[-1])
            y_pred, y_true = np.round(np.median(ppc_samples, axis=0)).astype(int), df_test['note'].values

            # 4. Calculate metrics
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='macro')
            mae = mean_absolute_error(y_true, y_pred)

            print(f"   -> [Scores Fold {fold + 1}] Accuracy: {acc:.3f} | F1 Macro: {f1:.3f} | MAE: {mae:.3f}")
            raw_results[model_name].append({'fold': fold + 1, 'accuracy': acc, 'f1_macro': f1, 'mae': mae})

    # 5. Synthesis and statistical aggregation of scores
    summary_rows = []
    for model_name, metrics in raw_results.items():
        metrics_df = pd.DataFrame(metrics)
        summary_rows.append({
            'Model': model_name, 'Accuracy (Mean)': metrics_df['accuracy'].mean(),
            'Accuracy (Std)': metrics_df['accuracy'].std(), 'F1 Macro (Mean)': metrics_df['f1_macro'].mean(),
            'F1 Macro (Std)': metrics_df['f1_macro'].std(), 'MAE (Mean)': metrics_df['mae'].mean(),
            'MAE (Std)': metrics_df['mae'].std(),
        })

    return pd.DataFrame(summary_rows)


def run_sampling(model, draws=1000, tune=1000):
    with model:
        idata = pm.sample(draws=draws, tune=tune, return_inferencedata=True)
        pm.compute_log_likelihood(idata)
        pm.sample_posterior_predictive(idata, extend_inferencedata=True)
    return idata


# ==========================================
# 5. PLOTS AND VISUALIZATIONS
# ==========================================
def plot_probabilistic_confusion_matrix(df_probs, name = ""):
    """
    Affiche une heatmap des probabilités moyennes allouées à chaque classe
    en fonction de la note réelle.
    """
    # 1. Calculer la moyenne des probabilités pour chaque groupe de note réelle
    prob_cols = [f"Prob_Note_{i}" for i in range(1, 6)]
    avg_probs = df_probs.groupby("Note_Reelle")[prob_cols].mean()

    # Renommer les axes pour plus de clarté
    avg_probs.columns = [f"Note {i}" for i in range(1, 6)]

    # 2. Dessiner la heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        avg_probs,
        annot=True,
        fmt=".2f",
        cmap="Greens",
        vmin=0, vmax=1,
        cbar_kws={'label': 'Probabilité moyenne'}
    )

    plt.title(f"Probabilities confusion matrix for {name}", fontsize=12,
              fontweight='bold')
    plt.ylabel("Note Réelle (Données d'origine)", fontsize=10)
    plt.xlabel("Classes de probabilités prédites", fontsize=10)
    plt.tight_layout()
    plt.show()

def plot_predictions_distribution(y_true, dict_preds, title_suffix="", save=False):
    data_list = [{'Source': 'Actual Values', 'Note': note} for note in y_true]
    for name, y_pred in dict_preds.items():
        data_list.extend([{'Source': name, 'Note': note} for note in y_pred])

    df_all = pd.DataFrame(data_list)
    fig, ax = plt.subplots(figsize=(12, 6))
    labels = [0, 1, 2, 3, 4]
    palette = sns.color_palette("deep", n_colors=len(dict_preds) + 1)

    sns.countplot(data=df_all, x='Note', hue='Source', order=labels, palette=palette, ax=ax)
    ax.set_title(f"Comparison of Grade Distribution - {title_suffix}", fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_xlabel("Grades", fontsize=11)
    ax.legend(title="Models / Actual", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

    for container in ax.containers:
        ax.bar_label(container, fmt='%d', padding=3, fontsize=8, rotation=0)

    plt.tight_layout()
    if save:
        os.makedirs("outputs/model_results/benchmark/plots/", exist_ok=True)
        plt.savefig(f"outputs/model_results/benchmark/plots/single_bar_dist_{title_suffix}.png", bbox_inches='tight')
    else:
        plt.show()


def plot_confusion_matrices(y_true, dict_preds, title_suffix="", save=False, nrows=2, ncols=3):
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 5.5))
    if nrows != 1 and ncols != 1: axes = axes.ravel()
    labels = [0, 1, 2, 3, 4]

    for idx, (name, y_pred) in enumerate(dict_preds.items()):
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=labels, yticklabels=labels, ax=axes[idx],
                    cbar=False)
        axes[idx].set_title(name, fontsize=12, fontweight='bold')
        axes[idx].set_ylabel("Actual Grades")
        axes[idx].set_xlabel("Predicted Grades")

    plt.suptitle(f"Confusion Matrices - {title_suffix}", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    if save:
        os.makedirs("outputs/model_results/benchmark/plots/", exist_ok=True)
        plt.savefig(f"outputs/model_results/benchmark/plots/confusion_matrix_{title_suffix}.png")
    else:
        plt.show()


def prior_predictive_checks(model, save=False, model_label="model"):
    save_path = f"outputs/model_results/MCLR/"
    with model:
        dt = pm.sample_prior_predictive(draws=500)
        pc = az.plot_ppc_dist(dt, group="prior_predictive")
        if save:
            plt.savefig(save_path + f"{model_label}_prior_predictive_plot.png")
        else:
            plt.show()


def evaluate_and_save_results(model, idata, df, var_names, model_label="Model", save=True):
    print(f"\n === {model_label.upper()} RESULTS ===")
    save_path = f"outputs/model_results/MCLR/"

    print(az.summary(idata, var_names=var_names))
    az.plot_forest(idata, var_names=var_names, combined=True)
    if save:
        plt.savefig(save_path + f"{model_label}_forest_plot.png")
    else:
        plt.show()

    random_effects = ['v_photo'] if 'u_personne' not in idata.posterior else ['u_personne', 'v_photo']
    az.plot_forest(idata, var_names=random_effects, combined=True)
    if save:
        plt.savefig(save_path + f"{model_label}_random_features_forest_plot.png")
    else:
        plt.show()

    tableau_resultats = az.summary(idata, var_names=var_names + random_effects)
    tableau_resultats.to_excel(f"outputs/model_results/MCLR/{model_label}_beta_coef.xlsx")
    tableau_seuils = az.summary(idata, var_names=["cutpoints"])
    tableau_seuils.to_excel(f"outputs/model_results/MCLR/{model_label}_seuils.xlsx")

    print(f"Total number of divergences: {idata.sample_stats['diverging'].sum().item()}")

    trace_vars = var_names + ["sigma_photo"] + (["sigma_personne"] if "sigma_personne" in idata.posterior else [])
    az.plot_trace_dist(idata, var_names=trace_vars)
    if save:
        plt.savefig(save_path + f"{model_label}_trace_plot.png")
    else:
        plt.show()

    with model:
        pm.sample_posterior_predictive(idata, extend_inferencedata=True)

    az.plot_ppc_dist(idata)
    if save:
        plt.savefig(save_path + f"{model_label}_ppc_dist_plot.png")
    else:
        plt.show()

    az.plot_ppc_pava(idata, data_type="categorical")
    if save:
        plt.savefig(save_path + f"{model_label}_ppc_pava_plot.png")
    else:
        plt.show()

    az.plot_rank(idata, var_names=var_names)
    if save:
        plt.savefig(save_path + f"{model_label}_rank_plot.png")
    else:
        plt.show()

    ppc_samples = idata.posterior_predictive["y_obs"].values.reshape(-1,
                                                                     idata.posterior_predictive["y_obs"].values.shape[
                                                                         -1])
    y_pred, y_true = np.round(np.median(ppc_samples, axis=0)).astype(int), df['note'].values
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[1, 2, 3, 4, 5], yticklabels=[1, 2, 3, 4, 5])
    plt.ylabel('Actual Values (Data)')
    plt.xlabel('Predicted Values (Model)')
    plt.title(f'Confusion Matrix - {model_label} (Median of Predictions)')
    if save:
        plt.savefig(save_path + f"{model_label}_confusion_matrix.png")
    else:
        plt.show()

    print(f"\n === CLASSIFICATION REPORT ({model_label.upper()}) ===")
    print(classification_report(y_true, y_pred))

    print(f"\n === LOO-CV ({model_label.upper()}) ===")
    loo_result = az.loo(idata)
    print(loo_result)
    az.plot_khat(loo_result)
    if save:
        plt.savefig(save_path + f"{model_label}_khat_plot.png")
    else:
        plt.show()