import pymc as pm
import numpy as np
import pandas as pd
import arviz as az


def run_model(df_sondage, n_pentes, n_vitesses, n_verts, n_types):
    """
    Définit le modèle logistique ordonné et lance l'échantillonnage MCMC.
    """
    with pm.Model() as model:
        # 1. Conteneurs de données (Permet de changer les valeurs plus tard pour prédire)
        pente_shared = pm.Data("pente_data", df_sondage['pente_idx'].values)
        vitesse_shared = pm.Data("vitesse_data", df_sondage['vitesse_idx'].values)
        verts_shared = pm.Data("verts_data", df_sondage['verts_idx'].values)
        type_shared = pm.Data("type_data", df_sondage['type_idx'].values)

        # 2. Les Priors (Vecteurs de Bonus/Malus pour chaque catégorie)
        beta_pente = pm.Normal("beta_pente", mu=0, sigma=1, shape=n_pentes)
        beta_vitesse = pm.Normal("beta_vitesse", mu=0, sigma=1, shape=n_vitesses)
        beta_verts = pm.Normal("beta_verts", mu=0, sigma=1, shape=n_verts)
        beta_type = pm.Normal("beta_type", mu=0, sigma=1, shape=n_types)

        # 3. L'équation du score d'élan (Combinaison linéaire)
        eta = (
                beta_pente[pente_shared]
                + beta_vitesse[vitesse_shared]
                + beta_verts[verts_shared]
                + beta_type[type_shared]
        )

        # 4. Les 4 seuils de jauge pour séparer les 5 étoiles
        cutpoints = pm.Normal(
            "cutpoints",
            mu=np.array([-2, -0.5, 0.5, 2]),
            sigma=2,
            transform=pm.distributions.transforms.ordered,
            shape=4
        )

        # 5. La Vraisemblance (Confrontation avec les vraies notes du sondage)
        Y_obs = pm.OrderedLogistic(
            "Y_obs",
            eta=eta,
            cutpoints=cutpoints,
            observed=df_sondage['note_idx'].values
        )

        # 6. Échantillonnage
        print("Lancement de l'échantillonnage PyMC...")
        idata = pm.sample(draws=1000, tune=1000, return_inferencedata=True)

    return model, idata


if __name__ == "__main__":
    # --- Préparation des données de test ---
    np.random.seed(42)
    N = 300

    # Nombre de catégories pour chaque critère
    n_pentes = 3  # 0: Faible, 1: Moyenne, 2: Forte
    n_vitesses = 2  # 0: Zone 30, 1: Zone 50
    n_verts = 3  # 0: Aucun, 1: Quelques-uns, 2: Beaucoup
    n_types = 2  # 0: Bande, 1: Piste séparée

    # Simulation d'un faux tableau de sondage
    df_sondage = pd.DataFrame({
        'pente_idx': np.random.choice(n_pentes, N),
        'vitesse_idx': np.random.choice(n_vitesses, N),
        'verts_idx': np.random.choice(n_verts, N),
        'type_idx': np.random.choice(n_types, N),
        'note_idx': np.random.choice(5, N)  # Notes de 0 à 4 (pour 1 à 5 étoiles)
    })

    # --- Exécution sécurisée du modèle ---
    modele, idata = run_model(df_sondage, n_pentes, n_vitesses, n_verts, n_types)

    # --- Exemple d'affichage des résultats ---
    print("\nRésumé des coefficients de ton sondage :")
    print(az.summary(idata, var_names=["beta_pente", "beta_verts"]))