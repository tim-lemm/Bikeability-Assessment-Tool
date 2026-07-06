import numpy as np
import pandas as pd
import pymc as pm
import pymc.distributions.transforms as tr
import arviz as az
from matplotlib import pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, mean_absolute_error
import seaborn as sns
from utils_survey_analysis import import_and_merge_data_base
import xarray as xr


def load_and_preprocess_data(base_path, personnes_path, photos_path, drop = "first"):
    """
    Loads survey data and applies preprocessing steps including categorical encoding,
    numerical scaling, and calculating dimension sizes for PyMC models.

    Parameters:
    -----------
    base_path : str
        Path to the base survey CSV file.
    personnes_path : str
        Path to the people data CSV file.
    photos_path : str
        Path to the photos data CSV file.

    Returns:
    --------
    df : pd.DataFrame
        Preprocessed DataFrame.
    dims : dict
        Dictionary containing unique category counts (sizes) for model parameters.
    coords : dict
        Coordinate mapping for specific categories used in PyMC dimensions.
    """
    df = import_and_merge_data_base(base_path, personnes_path, photos_path)
    df['id_personne'] = df['id_personne'].astype('category').cat.codes
    df['id_photo'] = df['id_photo'].astype('category').cat.codes
    cat_cols = ['age', 'gender', 'job', 'electric_bike', 'speed', 'slope', 'green', 'type',
                'bike_use_frequency','bike_ownership', 'nbr_lane']
    # Encoding categorical features
    encoder_cat = OneHotEncoder(dtype=np.int64, drop = drop)
    enc_output = encoder_cat.fit_transform(df[cat_cols]).toarray()

    encoded_cols = encoder_cat.get_feature_names_out(cat_cols)
    df_enc = pd.DataFrame(enc_output, columns=encoded_cols, index=df.index)
    df_enc['id_personne'] = df['id_personne']
    df_enc['id_photo'] = df['id_photo']

    # Map target scale from 1-5 to 0-4
    df_enc['note'] = df['note'].values - 1

    # Extract dimension sizes
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

    return df_enc, dims, coords, cat_cols


def build_model_1(encoded_df, dims, cat_cols):
    """
    Builds the first PyMC Ordered Logistic model containing both
    individual socio-demographic features and photo attributes (Full Model).
    """

    X_age = encoded_df[[col for col in encoded_df.columns if col.startswith('age_')]].values
    X_gender = encoded_df[[col for col in encoded_df.columns if col.startswith('gender_')]].values
    X_job = encoded_df[[col for col in encoded_df.columns if col.startswith('job_')]].values
    X_electric_bike = encoded_df[[col for col in encoded_df.columns if col.startswith('electric_bike_')]].values
    X_bike_use_frequency = encoded_df[
        [col for col in encoded_df.columns if col.startswith('bike_use_frequency_')]].values
    X_bike_ownership = encoded_df[[col for col in encoded_df.columns if col.startswith('bike_ownership_')]].values

    X_nbr_lanes = encoded_df[[col for col in encoded_df.columns if col.startswith('nbr_lane_')]].values
    X_type = encoded_df[[col for col in encoded_df.columns if col.startswith('type_')]].values
    X_slope = encoded_df[[col for col in encoded_df.columns if col.startswith('slope_')]].values
    X_speed = encoded_df[[col for col in encoded_df.columns if col.startswith('speed_')]].values
    X_green = encoded_df[[col for col in encoded_df.columns if col.startswith('green_')]].values

    with pm.Model() as model:
        # Cutpoints
        cutpoints = pm.Normal(
            'cutpoints',
            mu=np.linspace(-2, 2, dims["n_notes"] - 1),
            sigma=1,
            transform=tr.ordered,
            shape=dims["n_notes"] - 1
        )

        # Fixed parameters (Socio-demographic)
        beta_age = pm.Normal("beta_age", mu=0, sigma=1, shape=dims["n_ages"]-1)
        beta_gender = pm.Normal("beta_gender", mu=0, sigma=1, shape=dims["n_genders"] - 1)
        beta_job = pm.Normal("beta_job", mu=0, sigma=1, shape=dims["n_jobs"] - 1)
        beta_electric_bike = pm.Normal("beta_electric_bike", mu=0, sigma=1, shape=dims["n_electric_bikes"] - 1)
        beta_bike_use_frequency = pm.Normal("beta_bike_use_frequency", mu=0, sigma=1, shape=dims["n_bike_use_frequency"] - 1)
        beta_bike_ownership = pm.Normal("beta_bike_ownership", mu=0, sigma=1, shape=dims["n_bike_ownership"] - 1)

        # Fixed parameters (Photo attributes)
        beta_nbr_lanes = pm.Normal("beta_nbr_lanes", mu=0, sigma=1, shape = dims["n_nbr_lanes"] - 1)
        beta_type = pm.Normal("beta_type", mu=0, sigma=1, shape=dims["n_types"] - 1)
        beta_slope = pm.Normal("beta_slope", mu=0, sigma=1, shape=dims["n_slopes"] - 1)
        beta_speed = pm.Normal("beta_speed", mu=0, sigma=1, shape=dims["n_speeds"] - 1)
        beta_green = pm.Normal("beta_green", mu=0, sigma=1, shape=dims["n_greens"] - 1)

        # Random effects
        sigma_personne = pm.HalfNormal("sigma_personne", sigma=1)
        u_personne = pm.Normal("u_personne", mu=0, sigma=sigma_personne, shape=dims["n_personne"])

        sigma_photo = pm.HalfNormal("sigma_photo", sigma=1)
        v_photo = pm.Normal("v_photo", mu=0, sigma=sigma_photo, shape=dims["n_photos"])

        # Latent variable linear combination
        mu = (
                u_personne[encoded_df['id_personne'].values] +
                pm.math.dot(X_age, beta_age) +
                pm.math.dot(X_gender, beta_gender) +
                pm.math.dot(X_job, beta_job) +
                pm.math.dot(X_electric_bike, beta_electric_bike) +
                pm.math.dot(X_bike_use_frequency, beta_bike_use_frequency) +
                pm.math.dot(X_bike_ownership, beta_bike_ownership) +

                v_photo[encoded_df['id_photo'].values] +
                pm.math.dot(X_nbr_lanes, beta_nbr_lanes) +
                pm.math.dot(X_type, beta_type) +
                pm.math.dot(X_slope, beta_slope) +
                pm.math.dot(X_speed, beta_speed) +
                pm.math.dot(X_green, beta_green)
        )

        pm.OrderedLogistic("y_obs", eta=mu, cutpoints=cutpoints, observed=encoded_df['note'].values)

    return model

def build_model_2(encoded_df, dims, coords):
    """
    Builds the second PyMC Ordered Logistic model containing only
    photo attributes and context random effects (Restricted Model).
    """
    with pm.Model() as model_2:
        # Cutpoints
        cutpoints = pm.Normal(
            'cutpoints',
            mu=np.linspace(-2, 2, dims["n_notes"] - 1),
            sigma=1,
            transform=tr.ordered,
            shape=dims["n_notes"] - 1
        )
        X_nbr_lanes = encoded_df[[col for col in encoded_df.columns if col.startswith('nbr_lane_')]].values
        X_type = encoded_df[[col for col in encoded_df.columns if col.startswith('type_')]].values
        X_slope = encoded_df[[col for col in encoded_df.columns if col.startswith('slope_')]].values
        X_speed = encoded_df[[col for col in encoded_df.columns if col.startswith('speed_')]].values
        X_green = encoded_df[[col for col in encoded_df.columns if col.startswith('green_')]].values

        # Fixed parameters (Photo attributes)
        beta_nbr_lanes = pm.Normal("beta_nbr_lanes", mu=0, sigma=1, shape=dims["n_nbr_lanes"] - 1)
        beta_type = pm.Normal("beta_type", mu=0, sigma=1, shape=dims["n_types"] - 1)
        beta_slope = pm.Normal("beta_slope", mu=0, sigma=1, shape=dims["n_slopes"] - 1)
        beta_speed = pm.Normal("beta_speed", mu=0, sigma=1, shape=dims["n_speeds"] - 1)
        beta_green = pm.Normal("beta_green", mu=0, sigma=1, shape=dims["n_greens"] - 1)

        # Random effects
        sigma_photo = pm.HalfNormal("sigma_photo", sigma=1)
        v_photo = pm.Normal("v_photo", mu=0, sigma=sigma_photo, shape=dims["n_photos"])

        # Latent variable linear combination
        mu = (
                v_photo[encoded_df['id_photo'].values] +
                pm.math.dot(X_nbr_lanes, beta_nbr_lanes) +
                pm.math.dot(X_type, beta_type) +
                pm.math.dot(X_slope, beta_slope) +
                pm.math.dot(X_speed, beta_speed) +
                pm.math.dot(X_green, beta_green)
        )

        pm.OrderedLogistic("y_obs", eta=mu, cutpoints=cutpoints, observed=encoded_df['note'].values)

    return model_2

def run_sampling(model, draws=1000, tune=1000):
    """
    Samples from the posterior distribution of the provided PyMC model
    and computes log-likelihood values.

    Parameters:
    -----------
    model : pm.Model
        The compiled PyMC model object.
    draws : int
        Number of samples to draw.
    tune : int
        Number of iteration steps to tune the samplers.

    Returns:
    --------
    idata : az.InferenceData
        ArviZ InferenceData object containing posterior samples.
    """
    with model:
        idata = pm.sample(draws=draws, tune=tune, return_inferencedata=True)
        pm.compute_log_likelihood(idata)
    return idata

def prior_predictive_checks (model, save = False, model_label = "model"):
    save_path = f"outputs/model_results/MCLR/"
    with model:
        dt = pm.sample_prior_predictive(draws=500)
        pc = az.plot_ppc_dist(dt, group="prior_predictive")
        if save:
            plt.savefig(save_path + f"{model_label}_prior_predictive_plot.png")
        else:
            plt.show()


def evaluate_and_save_results(model, idata, df, var_names, model_label="Model", save = True):
    """
    Performs evaluation, prints logs, plots diagnostic graphics,
    and saves result tables to Excel files.

    Parameters:
    -----------
    model : pm.Model
        The compiled PyMC model object.
    idata : az.InferenceData
        The MCMC sampling inference results.
    df : pd.DataFrame
        The primary data table containing true labels.
    var_names : list
        List of main beta coefficients/variables to evaluate.
    results_prefix : str
        Prefix/Filename tag used to export excel summaries (e.g. "resultats_variables").
    model_label : str
        Text label for console logs distinction.
    """
    print(f"\n === {model_label.upper()} RESULTS ===")
    save_path = f"outputs/model_results/MCLR/"

    # Summaries
    summary = az.summary(idata, var_names=var_names)
    print(summary)

    # Forest Plots
    az.plot_forest(idata, var_names=var_names, combined=True)
    if save:
        plt.savefig(save_path + f"{model_label}_forest_plot.png")
    else :
        plt.show()

    random_effects = ['v_photo'] if 'u_personne' not in idata.posterior else ['u_personne', 'v_photo']
    az.plot_forest(idata, var_names=random_effects, combined=True)
    if save:
        plt.savefig(save_path + f"{model_label}_random_features_forest_plot.png")
    else :
        plt.show()

    # Save to Excel
    tableau_resultats = az.summary(idata, var_names=var_names + random_effects)
    tableau_resultats.to_excel(f"outputs/model_results/MCLR/{model_label}_beta_coef.xlsx")

    tableau_seuils = az.summary(idata, var_names=["cutpoints"])
    tableau_seuils.to_excel(f"outputs/model_results/MCLR/{model_label}_seuils.xlsx")

    # Divergences check
    divergences = idata.sample_stats["diverging"].sum().item()
    print(f"Total number of divergences: {divergences}")

    # Trace distributions
    trace_vars = var_names + ["sigma_photo"] + (["sigma_personne"] if "sigma_personne" in idata.posterior else [])
    az.plot_trace_dist(idata, var_names=trace_vars)
    if save:
        plt.savefig(save_path + f"{model_label}_trace_plot.png")
    else :
        plt.show()

    # Posterior Predictive Checks
    with model:
        pm.sample_posterior_predictive(idata, extend_inferencedata=True)

    az.plot_ppc_dist(idata)
    if save:
        plt.savefig(save_path + f"{model_label}_ppc_dist_plot.png")
    else :
        plt.show()

    az.plot_ppc_pava(idata, data_type="categorical")
    if save:
        plt.savefig(save_path + f"{model_label}_ppc_pava_plot.png")
    else :
        plt.show()

    az.plot_rank(idata, var_names=var_names)
    if save:
        plt.savefig(save_path + f"{model_label}_rank_plot.png")
    else :
        plt.show()

    # Confusion Matrix
    ppc_samples = idata.posterior_predictive["y_obs"].values
    ppc_samples = ppc_samples.reshape(-1, ppc_samples.shape[-1])

    y_pred = np.round(np.median(ppc_samples, axis=0)).astype(int)
    y_true = df['note'].values
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=[1, 2, 3, 4, 5], yticklabels=[1, 2, 3, 4, 5]
    )
    plt.ylabel('Valeurs Réelles (Données)')
    plt.xlabel('Valeurs Prédites (Modèle)')
    plt.title(f'Matrice de Confusion - {model_label} (Médiane des Prédictions)')
    if save:
        plt.savefig(save_path + f"{model_label}_confusion_matrix.png")
    else :
        plt.show()

    # Classification Report
    print(f"\n === CLASSIFICATION REPORT ({model_label.upper()}) ===")
    print(classification_report(y_true, y_pred))

    # Leave-One-Out Cross-Validation
    print(f"\n === LOO-CV ({model_label.upper()}) ===")
    loo_result = az.loo(idata)
    print(loo_result)
    az.plot_khat(loo_result)
    if save:
        plt.savefig(save_path + f"{model_label}_khat_plot.png")
    else :
        plt.show()


def run_benchmark(df, dims, coords, model_factories, n_splits=5, draws=1000, tune=1000):
    """
    Exécute une validation croisée de type Monte Carlo (Train-Test Split 80/20 répété)
    sur l'ensemble des modèles fournis pour évaluer leur capacité prédictive.
    """
    print(f"\n⚡ DÉMARRAGE DU BENCHMARK (Train-Test Split 80/20 répété {n_splits} fois)...")

    rs = ShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)
    raw_results = {name: [] for name in model_factories.keys()}

    for fold, (train_idx, test_idx) in enumerate(rs.split(df)):
        print(f"\n🔄 --- FOLD {fold + 1} / {n_splits} ---")
        df_train = df.iloc[train_idx].copy()
        df_test = df.iloc[test_idx].copy()

        for model_name, model_builder in model_factories.items():
            print(f"▶️ Entraînement de [{model_name}]...")

            # 1. Ajustement sur le jeu d'entraînement
            train_model = model_builder(df_train, dims, coords)
            with train_model:
                idata_train = pm.sample(draws=draws, tune=tune, return_inferencedata=True, progressbar=False,
                                        random_seed=42)

            print(f"Prédiction de [{model_name}] sur le jeu de test...")

            posterior_ds = idata_train["posterior"].dataset
            posterior_cleaned = posterior_ds.drop_vars("y_obs_probs", errors="ignore")

            idata_posterior_only = xr.DataTree()
            idata_posterior_only["posterior"] = posterior_cleaned

            # 2. Reconstruction du modèle avec les données de test (748 observations)
            test_model = model_builder(df_test, dims, coords)
            with test_model:
                ppc = pm.sample_posterior_predictive(idata_posterior_only, progressbar=False, random_seed=42)

            # 3. Extraction et réduction des prédictions (Médiane des tirages)
            ppc_samples = ppc.posterior_predictive["y_obs"].values
            ppc_samples = ppc_samples.reshape(-1, ppc_samples.shape[-1])

            y_pred = np.round(np.median(ppc_samples, axis=0)).astype(int)
            y_true = df_test['note'].values

            # 4. Calcul des métriques
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='macro')
            mae = mean_absolute_error(y_true, y_pred)

            print(f"   ↳ [Scores Fold {fold + 1}] Accuracy: {acc:.3f} | F1 Macro: {f1:.3f} | MAE: {mae:.3f}")

            raw_results[model_name].append({
                'fold': fold + 1,
                'accuracy': acc,
                'f1_macro': f1,
                'mae': mae
            })

    # 5. Synthèse et agrégation statistique des scores
    summary_rows = []
    for model_name, metrics in raw_results.items():
        metrics_df = pd.DataFrame(metrics)
        summary_rows.append({
            'Modèle': model_name,
            'Accuracy (Mean)': metrics_df['accuracy'].mean(),
            'Accuracy (Std)': metrics_df['accuracy'].std(),
            'F1 Macro (Mean)': metrics_df['f1_macro'].mean(),
            'F1 Macro (Std)': metrics_df['f1_macro'].std(),
            'MAE (Mean)': metrics_df['mae'].mean(),
            'MAE (Std)': metrics_df['mae'].std(),
        })

    return pd.DataFrame(summary_rows)