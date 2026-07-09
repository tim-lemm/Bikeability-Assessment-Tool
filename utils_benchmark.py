import os
from sklearn.model_selection import train_test_split, LeaveOneOut, StratifiedKFold, StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import accuracy_score, mean_absolute_error, precision_score, confusion_matrix, balanced_accuracy_score, recall_score, f1_score, cohen_kappa_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier
)
from utils_MCLR import *
from scipy.stats import mode

def plot_predictions_distribution(y_true, dict_preds, title_suffix="", save=False):
    # 1. On prépare une liste pour stocker les données de chaque modèle
    data_list = []

    # On ajoute d'abord les valeurs réelles comme référence
    for note in y_true:
        data_list.append({'Source': 'Valeurs Réelles', 'Note': note})

    # On ajoute ensuite les prédictions de chaque modèle
    for name, y_pred in dict_preds.items():
        for note in y_pred:
            data_list.append({'Source': name, 'Note': note})

    # On transforme le tout en un grand DataFrame
    df_all = pd.DataFrame(data_list)

    # 2. Création de la figure (un seul graphique, donc plus besoin de nrows/ncols)
    fig, ax = plt.subplots(figsize=(12, 6))
    labels = [0, 1, 2, 3, 4]

    # Palette de couleurs : une couleur distincte par modèle + une pour le Réel
    # 'Deep' est une palette très lisible, mais vous pouvez changer
    palette = sns.color_palette("deep", n_colors=len(dict_preds) + 1)

    # 3. Tracé du diagramme en barres unique
    sns.countplot(
        data=df_all,
        x='Note',
        hue='Source',
        order=labels,
        palette=palette,
        ax=ax
    )

    # 4. Personnalisation du graphique
    ax.set_title(f"Comparaison de la Distribution des Notes - {title_suffix}", fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel("Nombre de valeurs (Comptage)", fontsize=11)
    ax.set_xlabel("Notes", fontsize=11)

    # Positionnement de la légende à l'extérieur pour ne pas cacher les barres
    ax.legend(title="Modèles / Réel", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

    # Optionnel : Afficher les chiffres au-dessus des barres
    # Attention, s'il y a beaucoup de modèles, enlever ces lignes pour éviter de surcharger le graphique
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', padding=3, fontsize=8, rotation=0)

    plt.tight_layout()

    if save:
        os.makedirs("outputs/model_results/benchmark/plots/", exist_ok=True)
        plt.savefig(f"outputs/model_results/benchmark/plots/single_bar_dist_{title_suffix}.png", bbox_inches='tight')
    else:
        plt.show()

def plot_confusion_matrices(y_true, dict_preds, title_suffix="", save=False, nrows=2, ncols=3):
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*6, nrows*5.5))
    if nrows != 1 and ncols != 1:
        axes = axes.ravel()

    labels = [0, 1, 2, 3, 4]

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
        os.makedirs("outputs/model_results/benchmark/plots/", exist_ok=True)
        plt.savefig(f"outputs/model_results/benchmark/plots/confusion_matrix_{title_suffix}.png")
    else:
        plt.show()

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
    prec_class_lists = {i:[] for i in range(num_class)}

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

def _split_data(X, y, train_idx, test_idx):
    if hasattr(X, "iloc"):
        return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def run_train_test_ML(models, X, y, test_size=0.2, random_state=42):
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,random_state=random_state)
    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)
    results = {}
    train_predictions_for_plots = {}
    test_predictions_for_plots = {}

    X_age_test = df_test[[col for col in df_test.columns if col.startswith('age_')]].values
    X_gender_test = df_test[[col for col in df_test.columns if col.startswith('gender_')]].values
    X_job_test = df_test[[col for col in df_test.columns if col.startswith('job_')]].values
    X_electric_bike_test = df_test[[col for col in df_test.columns if col.startswith('electric_bike_')]].values
    X_bike_use_frequency_test = df_test[
        [col for col in df_test.columns if col.startswith('bike_use_frequency_')]].values
    X_bike_ownership_test = df_test[[col for col in df_test.columns if col.startswith('bike_ownership_')]].values

    X_nbr_lanes_test = df_test[[col for col in df_test.columns if col.startswith('nbr_lane_')]].values
    X_type_test = df_test[[col for col in df_test.columns if col.startswith('type_')]].values
    X_slope_test = df_test[[col for col in df_test.columns if col.startswith('slope_')]].values
    X_speed_test = df_test[[col for col in df_test.columns if col.startswith('speed_')]].values
    X_green_test = df_test[[col for col in df_test.columns if col.startswith('green_')]].values

    id_personne_test = df_test['id_personne'].values
    id_photo_test = df_test['id_photo'].values

    models = get_models_MCLR(df_train, dims)
    for name, model in models.items():
        model_actif = models[name][0]
        # print("Conteneurs de données enregistrés :", [v.name for v in model_actif.data_vars])
        with model_actif:
            trace = pm.sample(draws=1000, tune=1000, return_inferencedata=True)
            idata_train = pm.sample_posterior_predictive(trace)
            if name == "MCLR Model 1":
                pm.set_data({
                    "X_age": X_age_test,
                    "X_gender": X_gender_test,
                    "X_job": X_job_test,
                    "X_electric_bike": X_electric_bike_test,
                    "X_bike_use_frequency": X_bike_use_frequency_test,
                    "X_bike_ownership": X_bike_ownership_test,
                    "X_nbr_lanes": X_nbr_lanes_test,
                    "X_type": X_type_test,
                    "X_slope": X_slope_test,
                    "X_speed": X_speed_test,
                    "X_green": X_green_test,
                    "id_personne": id_personne_test,
                    "id_photo": id_photo_test,
                    "y_obs_data":y_test
                })
            else :
                pm.set_data({
                    "X_nbr_lanes": X_nbr_lanes_test,
                    "X_type": X_type_test,
                    "X_slope": X_slope_test,
                    "X_speed": X_speed_test,
                    "X_green": X_green_test,
                    "id_photo": id_photo_test,
                    "y_obs_data": y_test
                })
            idata_test = pm.sample_posterior_predictive(trace)

        train_metrics = get_metrics_MCLR(y_train, idata_train)
        test_metrics = get_metrics_MCLR(y_test, idata_test)

        results[name] = {}
        for k, v in train_metrics.items():
            results[name][f'Train {k}'] = v
        for k, v in test_metrics.items():
            results[name][f'Test {k}'] = v

        y_pred_train_samples = idata_train["posterior_predictive"]["y_obs"].values
        y_pred_train_flat = y_pred_train_samples.reshape(-1, y_pred_train_samples.shape[-1])
        train_predictions_for_plots[name] = mode(y_pred_train_flat, axis=0, keepdims=True).mode[0]

        y_pred_test_samples = idata_test["posterior_predictive"]["y_obs"].values
        y_pred_test_flat = y_pred_test_samples.reshape(-1, y_pred_test_samples.shape[-1])
        test_predictions_for_plots[name] = mode(y_pred_test_flat, axis=0, keepdims=True).mode[0]

    return pd.DataFrame(results).T, train_predictions_for_plots, test_predictions_for_plots

def run_shufflesplit_ML(models, X, y, n_splits=5, test_size=0.2, random_state=42):
    cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
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

            # Calcul des metriques d'entrainement et de test pour le fold actuel
            train_metrics = get_metrics_ML(y_train, train_pred)
            test_metrics = get_metrics_ML(y_test, test_pred)

            # Combinaison des metriques avec des prefixes clairs
            combined_metrics = {}
            for k, v in train_metrics.items():
                combined_metrics[f'Train {k}'] = v
            for k, v in test_metrics.items():
                combined_metrics[f'Test {k}'] = v

            fold_metrics.append(combined_metrics)

        df_folds = pd.DataFrame(fold_metrics)
        results[name] = df_folds.mean().to_dict()
        dict_preds_train[name] = np.array(all_train_preds)
        dict_preds_test[name] = np.array(all_test_preds)

    return pd.DataFrame(results).T, dict_preds_train, y_train_concat, dict_preds_test, y_test_concat


def run_shufflesplit_MCLR(df, dims, n_splits=5, test_size=0.2, random_state=42):
    # Séparation des features et de la cible à partir du DataFrame global
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    models = get_models_MCLR(df, dims)
    cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    results = {}
    dict_preds_train = {}
    dict_preds_test = {}

    splits = list(cv.split(X, y))
    y_train_concat = []
    y_test_concat = []

    # 1. Accumulation des vraies valeurs cibles à travers tous les folds
    for train_idx, test_idx in splits:
        y_train_concat.extend(y.iloc[train_idx].values)
        y_test_concat.extend(y.iloc[test_idx].values)

    y_train_concat = np.array(y_train_concat)
    y_test_concat = np.array(y_test_concat)

    # Fonction utilitaire interne pour extraire le dictionnaire complet requis par pm.set_data
    def _get_pymc_data_dict(df_fold, y_fold=None):
        data_dict = {
            "X_nbr_lanes": df_fold[[col for col in df_fold.columns if col.startswith('nbr_lane_')]].values,
            "X_type": df_fold[[col for col in df_fold.columns if col.startswith('type_')]].values,
            "X_slope": df_fold[[col for col in df_fold.columns if col.startswith('slope_')]].values,
            "X_speed": df_fold[[col for col in df_fold.columns if col.startswith('speed_')]].values,
            "X_green": df_fold[[col for col in df_fold.columns if col.startswith('green_')]].values,
            "id_photo": df_fold['id_photo'].values
        }
        # Si c'est le Modèle 1, on ajoute les variables socio-démographiques
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

    # 2. Boucle sur les modèles (votre dictionnaire de modèles PyMC)
    for name, model_info in models.items():
        model_actif = model_info[0]

        fold_metrics = []
        all_train_preds = []
        all_test_preds = []

        print(f"-> Entraînement et évaluation en Cross-Validation de : {name}")

        for split_idx, (train_idx, test_idx) in enumerate(splits):
            print(f"   Fold {split_idx + 1}/{n_splits}...")

            df_train = df.iloc[train_idx]
            df_test = df.iloc[test_idx]

            y_train = df_train.iloc[:, -1]
            y_test = df_test.iloc[:, -1]

            with model_actif:
                # --- ÉTAPE A : CONFIGURATION TRAIN & SAMPLING ---
                train_data_dict = _get_pymc_data_dict(df_train, y_train)
                pm.set_data(train_data_dict)

                trace = pm.sample(draws=1000, tune=1000, return_inferencedata=True, progressbar=False)
                idata_train = pm.sample_posterior_predictive(trace, progressbar=False)

                # --- ÉTAPE B : EXTRACTION PRÉDICTIONS TRAIN (MODE) ---
                y_pred_train_samples = idata_train["posterior_predictive"]["y_obs"].values
                y_pred_train_flat = y_pred_train_samples.reshape(-1, y_pred_train_samples.shape[-1])
                train_pred = mode(y_pred_train_flat, axis=0, keepdims=True).mode[0]
                all_train_preds.extend(train_pred)

                # --- ÉTAPE C : CONFIGURATION TEST & POSTERIOR PREDICTIVE ---
                test_data_dict = _get_pymc_data_dict(df_test, y_test)
                pm.set_data(test_data_dict)

                idata_test = pm.sample_posterior_predictive(trace, progressbar=False)

                # --- ÉTAPE D : EXTRACTION PRÉDICTIONS TEST (MODE) ---
                y_pred_test_samples = idata_test["posterior_predictive"]["y_obs"].values
                y_pred_test_flat = y_pred_test_samples.reshape(-1, y_pred_test_samples.shape[-1])
                test_pred = mode(y_pred_test_flat, axis=0, keepdims=True).mode[0]
                all_test_preds.extend(test_pred)

            # Calcul des métriques bayésiennes globales (moyennées sur l'échantillonnage posterior)
            train_metrics = get_metrics_MCLR(y_train, idata_train)
            test_metrics = get_metrics_MCLR(y_test, idata_test)

            # Combinaison des métriques avec préfixes
            combined_metrics = {}
            for k, v in train_metrics.items():
                combined_metrics[f'Train {k}'] = v
            for k, v in test_metrics.items():
                combined_metrics[f'Test {k}'] = v

            fold_metrics.append(combined_metrics)

        # 3. Agrégation des résultats du modèle actuel
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

        results[name] = get_metrics_ML(y, all_test_preds)
        dict_preds_train[name] = np.array(all_train_preds)
        dict_preds_test[name] = all_test_preds

    return pd.DataFrame(results).T, dict_preds_train, y_train_concat, dict_preds_test, y

def load_and_prepare_data(base_csv, personnes_csv, photos_csv):
    df, _, _ = load_and_preprocess_data(base_csv, personnes_csv, photos_csv, drop = None)
    # print(df.head().to_string())

    liste_cat = [f"nbr_lane_{i}" for i in range(4)] + [f"speed_{i}"for i in range(4)] + [f"slope_{i}" for i in range(3)] + [f"green_{i}" for i in range(3)] + [f"type_{i}" for i in range(4)]

    X = df[liste_cat].copy()
    y = df["note"].astype(int)

    df, dims, coords = load_and_preprocess_data(base_csv, personnes_csv, photos_csv, drop="first")

    return X, y, df, dims, coords

def get_models_ML():
    return {
        "Classification Tree": DecisionTreeClassifier(random_state=42),
        "Gradient Boost": GradientBoostingClassifier(random_state=42, learning_rate=0.2,
                                                     max_depth=5,
                                                     min_samples_split=2,
                                                     min_samples_leaf=1,
                                                     n_estimators=500,
                                                     subsample=0.8),
        "HistGradient Boost": HistGradientBoostingClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Ada Boost": AdaBoostClassifier(random_state=42),
        "ExtraTrees": ExtraTreesClassifier(random_state=42)
    }

def get_models_MCLR(df, dims):
    return {
        "MCLR Model 1": [build_model_1(df, dims), ["beta_age", "beta_gender", "beta_job", "beta_electric_bike", "beta_bike_use_frequency", "beta_bike_ownership",
        "beta_nbr_lanes", "beta_type", "beta_slope", "beta_speed", "beta_green"]],
        "MCLR Model 2": [build_model_2(df, dims), ["beta_nbr_lanes", "beta_type", "beta_slope", "beta_speed", "beta_green"]]
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

def optimize_model_hp(X, y, base_model, param_grid, scoring_metric="balanced_accuracy"):
    # 1. Définition du modèle de base
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # 2. Définition de la grille d'hyperparamètres à tester

    print("Lancement de la recherche des hyperparamètres optimaux (GridSearchCV)...")

    # 4. Configuration du GridSearch (ici en 5-fold cross-validation)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring=scoring_metric,
        cv=cv,
        n_jobs=-1,
        verbose=3,
        return_train_score=True
    )

    # 5. Exécution de la recherche
    grid_search.fit(X, y)

    results_df = pd.DataFrame(grid_search.cv_results_)

    cols_to_keep = ['params', 'mean_train_score', 'mean_test_score']
    print(results_df[cols_to_keep].sort_values(by='mean_test_score', ascending=False))

    # --- Affichage des meilleurs résultats ---
    print("\n--- Résultats de l'optimisation ---")
    print(f"Meilleur score ({scoring_metric}) : {grid_search.best_score_}")
    print("Meilleurs hyperparamètres trouvés :", grid_search.best_params_)

    best_index = grid_search.best_index_
    print(f"\nScore Train du meilleur modèle : {results_df.loc[best_index, 'mean_train_score']}")
    print(f"Score Test du meilleur modèle  : {results_df.loc[best_index, 'mean_test_score']}")

    return grid_search.best_estimator_, grid_search.best_params_