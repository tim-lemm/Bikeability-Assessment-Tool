import numpy as np
import pandas as pd
import pymc as pm
import pymc.distributions.transforms as tr
import arviz as az
from matplotlib import pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from utils_survey_analysis import import_and_merge_data_base
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

#real data

df = import_and_merge_data_base("data/survey/data-base.csv", "data/survey/data-personnes.csv", "data/survey/data-photos.csv")

### === PREPROCESSING ===

print("\n === DATA PREPROCESSING ===")
#encoding of categorical features

cat_cols = ['id_personne', 'id_photo', 'age', 'gender','job', 'electric_bike', 'speed', 'slope', 'green', 'type']
encoder_cat = OrdinalEncoder(dtype=np.int64)

df[[f"{col}_idx" for col in cat_cols]] = encoder_cat.fit_transform(df[cat_cols])

#encoding of numerical features

num_cols = ['nbr_lane']
scaler_num = StandardScaler()

df[[f"{col}_scaled" for col in num_cols]] = scaler_num.fit_transform(df[num_cols])

# note form 1 - 5 to 0 - 4

df['note_idx'] = df['note'] - 1

# size of values

n_personne = df['id_personne_idx'].nunique()
n_genders = df['gender_idx'].nunique()
n_ages = df['age_idx'].nunique()
n_jobs = df['job_idx'].nunique()
n_electric_bikes = df['electric_bike_idx'].nunique()

n_photos = df['id_photo_idx'].nunique()
n_types = df['type_idx'].nunique()
n_speeds = df['speed_idx'].nunique()
n_slopes = df['slope_idx'].nunique()
n_greens = df['green_idx'].nunique()

n_notes = 5

noms_separation = list(encoder_cat.categories_[2])
noms_gender = list(encoder_cat.categories_[3])
coordonnees = {
    "categories_separation": noms_separation,
    "categories_gender": noms_gender
}

### === Model construction ===

print("\n === MODEL CONSTRUCTION ===")

with pm.Model() as model:
    # Seuils
    cutpoints = pm.Normal(
        'cutpoints',
        mu=np.linspace(-2,2, n_notes-1 ),
        sigma=1,
        transform=tr.ordered,
        shape=n_notes-1
    )

    # fixed parameters

    beta_age = pm.Normal("beta_age", mu=0, sigma=1, shape=n_ages)
    beta_gender = pm.Normal("beta_gender", mu=0, sigma=1, shape=n_genders, dims="categories_gender")
    beta_job = pm.Normal("beta_job", mu=0, sigma=1, shape=n_jobs)
    beta_electric_bike = pm.Normal("beta_electric_bike", mu=0, sigma=1, shape=n_electric_bikes)

    beta_nbr_lanes = pm.Normal("beta_nbr_lanes", mu=0, sigma=1)
    beta_type = pm.Normal("beta_type", mu=0, sigma=1, shape=n_types, dims="categories_separation")
    beta_slope = pm.Normal("beta_slope", mu=0, sigma=1, shape=n_slopes)
    beta_speed = pm.Normal("beta_speed", mu=0, sigma=1, shape=n_speeds)
    beta_green = pm.Normal("beta_green", mu=0, sigma=1, shape=n_greens)

    # random parameters
    sigma_personne = pm.HalfNormal("sigma_personne", sigma=1)
    u_personne = pm.Normal("u_personne", mu=0, sigma=sigma_personne, shape=n_personne)

    sigma_photo = pm.HalfNormal("sigma_photo", sigma=1)
    v_photo = pm.Normal("v_photo", mu=0, sigma=sigma_photo, shape=n_photos)

    # latent variable
    mu = (
            u_personne[df['id_personne_idx'].values] +
            beta_age[df['age_idx'].values] +
            beta_gender[df['gender_idx'].values] +
            beta_job[df['job_idx'].values] +
            beta_electric_bike[df['electric_bike_idx'].values] +

            v_photo[df['id_photo_idx'].values] +
            beta_nbr_lanes * df['nbr_lane_scaled'].values +
            beta_type[df['type_idx'].values] +
            beta_slope[df['slope_idx'].values] +
            beta_speed[df['speed_idx'].values] +
            beta_green[df['green_idx'].values]
    )

    y_obs = pm.OrderedLogistic("y_obs", eta=mu, cutpoints=cutpoints, observed=df['note_idx'].values)

    idata = pm.sample(draws=1000, tune=1000, return_inferencedata=True)
    pm.compute_log_likelihood(idata)

### === MODEL RESULTS ===

print("\n === MODEL RESULTS ===")
variables_a_afficher = ["beta_age", "beta_gender", "beta_job", "beta_electric_bike", "beta_nbr_lanes", "beta_type", "beta_slope", "beta_speed", "beta_green"]

summary = az.summary(idata, var_names=variables_a_afficher)
print(summary)

summary_random = az.summary(idata, var_names=variables_a_afficher)
print(summary_random)

az.plot_forest(
    idata,
    var_names=variables_a_afficher,
    combined=True,
)
plt.show()

az.plot_forest(
    idata,
    var_names=['u_personne','v_photo'],
    combined=True,
)
plt.show()

tableau_resultats = az.summary(idata, var_names=variables_a_afficher+['u_personne','v_photo'])
tableau_resultats.to_excel("resultats_variables.xlsx")

tableau_seuils = az.summary(idata, var_names=["cutpoints"])
tableau_seuils.to_excel("seuils.xlsx")

# 1. Vérifier s'il y a eu des divergences lors du sampling
divergences = idata.sample_stats["diverging"].sum().item()
print(f"Nombre total de divergences : {divergences}")

# 2. Afficher un résumé des warnings de convergence
# ArviZ affiche automatiquement des alertes si R-hat ou ESS sont mauvais
print(az.summary(idata, var_names=variables_a_afficher))

# 3. Visualiser les Traces (Traceplot)

az.plot_trace_dist(idata, var_names=variables_a_afficher + ["sigma_personne", "sigma_photo"])
plt.show()

# 4. Posterior Predictive Checks

with model:
    # Générer des données à partir de la distribution postérieure
    pm.sample_posterior_predictive(idata, extend_inferencedata=True)

# Graphique de comparaison
az.plot_ppc_rootogram(idata)
plt.show()
az.plot_rank(idata, var_names=variables_a_afficher)

# 5. Confusion Matrix

# Extraire les prédictions du modèle (forme: chaines x tirages x observations)
ppc_samples = idata.posterior_predictive["y_obs"].values
ppc_samples = ppc_samples.reshape(-1, ppc_samples.shape[-1])

# Pour chaque observation, on prend la note médiane ou la plus fréquente (mode) prédite
y_pred = np.round(np.median(ppc_samples, axis=0)).astype(int)
y_true = df['note_idx'].values
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[1, 2, 3, 4, 5],
    yticklabels=[1, 2, 3, 4, 5]
)
plt.ylabel('Valeurs Réelles (Données)')
plt.xlabel('Valeurs Prédites (Modèle)')
plt.title('Matrice de Confusion (Médiane des Prédictions)')
plt.show()

# 6. rapport de classification complet (Précision, Recall, F1-score)
print("\n === RAPPORT DE CLASSIFICATION ===")
print(classification_report(y_true, y_pred))

# 7. LOO
print("\n === LOO-CV ===")
loo_result= az.loo(idata)
print(loo_result)
az.plot_khat(loo_result)
plt.show()
