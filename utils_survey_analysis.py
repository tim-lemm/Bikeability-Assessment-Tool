import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, make_scorer, cohen_kappa_score,classification_report
from sklearn import tree
from sklearn.model_selection import train_test_split, LeaveOneOut, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Chargement de la palette globale
cmap = plt.get_cmap("Set2")

def import_and_merge_data_base(CHEMIN_DATABASE, CHEMIN_PERSONNES, CHEMIN_DATAPHOTO):
    df_data_base = pd.read_csv(CHEMIN_DATABASE) #all responses (photo, respondent, grade)
    df_data_personnes = pd.read_csv(CHEMIN_PERSONNES)  #personal info about respondent
    df_data_photos = pd.read_csv(CHEMIN_DATAPHOTO) #photo (infra) characteristics

    df_merged = pd.merge(df_data_base, df_data_photos, on="id_photo")
    df_merged = pd.merge(df_merged, df_data_personnes, on="id_personne")
    return df_merged

def preprocess_data_for_mean_models (CHEMIN_SUMMARY, test_size = 0.2, random_state = 42):
    df_mean = pd.read_csv(CHEMIN_SUMMARY) #mean grade for each photo

    df_mean['Note (moyenne)'] = df_mean['Note (moyenne)'].str.replace(',', '.').astype(float)
    df_mean = df_mean[['Voies', 'Vitesse', 'Pente', 'Green', 'Type', 'Note (moyenne)']]
    df_mean.columns = ['nbr_lane', 'speed', 'slope', 'green', 'type', 'note']

    y_mean = df_mean['note']

    X_mean_classified = df_mean[['nbr_lane', 'speed', 'slope', 'green']]
    X_mean_type = df_mean[['type']]

    enc = OneHotEncoder(handle_unknown='ignore')
    X_mean_type_encoded = enc.fit_transform(X_mean_type)
    type_encoded_columns = enc.get_feature_names_out(['type'])

    X_mean = pd.concat([
        X_mean_classified.reset_index(drop=True),
        pd.DataFrame(X_mean_type_encoded.toarray(), columns=type_encoded_columns).reset_index(drop=True)
    ], axis=1)

    X_mean_train, X_mean_test, y_mean_train, y_mean_test = train_test_split(X_mean, y_mean, test_size=test_size, random_state=random_state)

    return X_mean_train, X_mean_test, y_mean_train, y_mean_test

def preprocess_data_for_all_models (CHEMIN_DATABASE, CHEMIN_PERSONNES, CHEMIN_DATAPHOTO, test_size = 0.2, random_state = 42):
    df_merged = import_and_merge_data_base(CHEMIN_DATABASE, CHEMIN_PERSONNES, CHEMIN_DATAPHOTO)
    y_all = df_merged['note']

    X_classified_df = df_merged[['nbr_lane', 'speed', 'slope', 'green']]
    X_type = df_merged[['type']]

    enc = OneHotEncoder(handle_unknown='ignore')
    X_type_encoded = enc.fit_transform(X_type)
    type_encoded_columns = enc.get_feature_names_out(['type'])

    X_all = pd.concat([
        X_classified_df.reset_index(drop=True),
        pd.DataFrame(X_type_encoded.toarray(), columns=type_encoded_columns).reset_index(drop=True)
    ], axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(X_all, y_all, test_size=test_size, random_state=random_state, stratify=y_all)

    return X_train, X_test, Y_train, Y_test

def execute_regression_mean(CHEMIN_SUMMARY, test_size = 0.2, random_state = 42):
    """Charge les données, les nettoie et exécute une régression OLS avec les valeurs moyennes des notes pour chaque photos."""

    X_train, X_test, y_train, y_test = preprocess_data_for_mean_models(CHEMIN_SUMMARY, test_size=test_size, random_state=random_state)

    #train linear regression mean model
    linear_mean_model = LinearRegression()
    linear_mean_model.fit(X_train, y_train)

    #predict values
    y_mean_lr_predicted_test = linear_mean_model.predict(X_test)
    y_mean_lr_predicted_train = linear_mean_model.predict(X_train)

    #compute all indicators
    mse_mean_lr_test = mean_squared_error(y_test, y_mean_lr_predicted_test)
    mse_mean_lr_train = mean_squared_error(y_train, y_mean_lr_predicted_train)
    r2_mean_lr_test = r2_score(y_test, y_mean_lr_predicted_test)
    r2_mean_lr_train = r2_score(y_train, y_mean_lr_predicted_train)

    #retrive coefficients
    coeffs_mean_lr = pd.Series(linear_mean_model.coef_, index=X_train.columns)
    coeffs_mean_lr = coeffs_mean_lr.sort_values(ascending=False)

    print("\nLinear Regression Mean - Model Results")
    print(f"\nTrain MSE: {mse_mean_lr_train}, R²: {r2_mean_lr_train}")
    print(f"Test MSE: {mse_mean_lr_test}, R²: {r2_mean_lr_test}")
    print("Coefficients:")
    print(coeffs_mean_lr)

    return linear_mean_model, coeffs_mean_lr

def execute_regression_all(CHEMIN_DATABASE, CHEMIN_PERSONNES, CHEMIN_DATAPHOTO, test_size = 0.2, random_state = 42):
    """Charge les données, les nettoie et exécute une régression OLS avec toutes les valeurs des résultats du sondage."""

    X_train, X_test, Y_train, Y_test = preprocess_data_for_all_models(CHEMIN_DATABASE, CHEMIN_PERSONNES, CHEMIN_DATAPHOTO, test_size, random_state)

    #train linear regression all model
    linear_all_model = LinearRegression()
    linear_all_model.fit(X_train, Y_train)

    #predict values
    y_all_lr_predicted_test = linear_all_model.predict(X_test)
    y_all_lr_predicted_train = linear_all_model.predict(X_train)

    #compute all indicators
    mse_all_lr_test = mean_squared_error(Y_test, y_all_lr_predicted_test)
    mse_all_lr_train = mean_squared_error(Y_train, y_all_lr_predicted_train)
    r2_all_lr_test = r2_score(Y_test, y_all_lr_predicted_test)
    r2_all_lr_train = r2_score(Y_train, y_all_lr_predicted_train)

    #retrive coefficients
    coeffs_all_lr = pd.Series(linear_all_model.coef_, index=X_train.columns)
    coeffs_all_lr = coeffs_all_lr.sort_values(ascending=False)

    print("\nLinear Regression All - Model Results")
    print(f"\nTrain MSE: {mse_all_lr_train}, R²: {r2_all_lr_train}")
    print(f"Test MSE: {mse_all_lr_test}, R²: {r2_all_lr_test}")
    print("Coefficients:")
    print(coeffs_all_lr)

    return linear_all_model, coeffs_all_lr

def execute_regression_tree_mean(CHEMIN_SUMMARY, DOSSIER_PHOTO, test_size = 0.2, random_state = 42):
    X_train, X_test, y_train, y_test = preprocess_data_for_mean_models(CHEMIN_SUMMARY,test_size=test_size,random_state=random_state)

    #train model
    regr_mean = DecisionTreeRegressor()
    regr_mean.fit(X_train, y_train)

    #predict values
    y_mean_predicted_train = regr_mean.predict(X_train)
    y_mean_predicted_test = regr_mean.predict(X_test)

    #compute all indicators
    mse_mean_test = mean_squared_error(y_test, y_mean_predicted_test)
    mse_mean_train = mean_squared_error(y_train, y_mean_predicted_train)
    mae_mean_train = mean_absolute_error(y_train, y_mean_predicted_train)
    mae_mean_test = mean_absolute_error(y_test, y_mean_predicted_test)
    r2_mean_train = r2_score(y_train, y_mean_predicted_train)
    r2_mean_test = r2_score(y_test, y_mean_predicted_test)
    depth_mean = regr_mean.get_depth()

    print("\nRegression Tree Mean - Model Results")
    print(f"\nTrain MSE: {mse_mean_train}, R²: {r2_mean_train}, mae: {mae_mean_train}")
    print(f"Test MSE: {mse_mean_test}, R²: {r2_mean_test}, mae: {mae_mean_test}")
    print(f"Depth: {depth_mean}")

    plt.figure(figsize=(15, 15))
    fig = plt.gcf()
    fig.patch.set_alpha(0.4)
    tree.plot_tree(regr_mean, feature_names=X_train.columns, filled=True)
    plt.title("Regression Mean")
    plt.savefig(f"{DOSSIER_PHOTO}model_results/regression_tree_mean.png")
    plt.close()
    return regr_mean

def execute_regression_tree_all(CHEMIN_DATABASE, CHEMIN_PERSONNES, CHEMIN_DATAPHOTO, DOSSIER_PHOTO, test_size = 0.2, random_state = 42, depth=None):
    X_train, X_test, y_train, y_test = preprocess_data_for_all_models(CHEMIN_DATABASE, CHEMIN_PERSONNES,
                                                                      CHEMIN_DATAPHOTO, test_size, random_state)

    #train model
    regr_all = DecisionTreeRegressor(max_depth=depth, random_state=random_state)
    regr_all.fit(X_train, y_train)

    # predict values
    Y_all_predicted_test = regr_all.predict(X_test)
    Y_all_predicted_train = regr_all.predict(X_train)

    #compute indicators
    mse_all_test = mean_squared_error(y_test, Y_all_predicted_test)
    r2_all_test = r2_score(y_test, Y_all_predicted_test)
    mae_all_test = mean_absolute_error(y_test, Y_all_predicted_test)

    mse_all_train = mean_squared_error(y_train, Y_all_predicted_train)
    r2_all_train = r2_score(y_train, Y_all_predicted_train)
    mae_all_train = mean_absolute_error(y_train, Y_all_predicted_train)

    depth_all = regr_all.get_depth()


    print("\nRegression Tree All - Model Results")
    print(f"Train MSE: {mse_all_train}, R²: {r2_all_train}, mae: {mae_all_train}")
    print(f"Test MSE: {mse_all_test}, R²: {r2_all_test}, mae: {mae_all_test}")
    print(f"Depth: {depth_all}")

    plt.figure(figsize=(30, 30))
    fig = plt.gcf()
    fig.patch.set_alpha(0.4)
    tree.plot_tree(regr_all, filled=True, feature_names=X_train.columns)
    plt.title("Regressor all")
    plt.savefig(f"{DOSSIER_PHOTO}model_results/regression_tree_all.png")
    plt.close()

def preprocess_data_for_ordered_model(CHEMIN_DATABASE, CHEMIN_PERSONNES, CHEMIN_DATAPHOTO, test_size = 0.2, random_state = 42):
    df_merged = import_and_merge_data_base(CHEMIN_DATABASE, CHEMIN_PERSONNES, CHEMIN_DATAPHOTO)

    df_merged_classified = df_merged[['nbr_lane', 'speed', 'slope', 'green','note']]
    df_merged_type = df_merged[["type"]]
    enc = OneHotEncoder(handle_unknown='ignore')
    df_merged_type_encoded = enc.fit_transform(df_merged_type)
    type_encoded_columns = enc.get_feature_names_out(['type'])
    df_merged = pd.concat([
        df_merged_classified.reset_index(drop=True),
        pd.DataFrame(df_merged_type_encoded.toarray(), columns = type_encoded_columns).reset_index(drop=True)
    ], axis = 1)

    order_grades = [1, 2, 3, 4, 5]
    df_merged["note"] = pd.Categorical(df_merged["note"], categories=order_grades, ordered=True)
    numerical_features = ["nbr_lane","speed",'slope','green']
    colonnes_type_encodes = [col for col in df_merged.columns if col.startswith('type')]
    features_type = colonnes_type_encodes[1:]
    all_features = features_type + numerical_features

    X_ordered = df_merged[all_features]
    y_ordered = df_merged["note"]
    X_train, X_test, y_train, y_test = train_test_split(X_ordered, y_ordered, test_size=test_size,
                                                        random_state=random_state, stratify=y_ordered)

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])

    return X_train_scaled, X_test_scaled, y_train, y_test

def execute_ordered_model(CHEMIN_DATABASE, CHEMIN_PERSONNES, CHEMIN_DATAPHOTO,test_size = 0.2, random_state = 42, distr = 'logit' ):
    X_train, X_test, y_train, y_test = preprocess_data_for_ordered_model(CHEMIN_DATABASE, CHEMIN_PERSONNES, CHEMIN_DATAPHOTO, test_size = test_size, random_state = random_state)

    model = OrderedModel(y_train, X_train, distr=distr)
    result = model.fit(method='bfgs', disp=False)

    print("\nOrdered Model - Model Results")
    print(result.summary())

    probabilites_test = result.predict(X_test)
    categories_notes = y_train.cat.categories
    predictions_index = np.argmax(probabilites_test.values, axis=1)
    y_pred = pd.Categorical.from_codes(predictions_index, categories=categories_notes, ordered=True)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy}")
    print(f"\nClassification Report: {classification_report(y_test, y_pred, zero_division=0)}")
    conf_matrix = confusion_matrix(y_test, y_pred, labels=categories_notes)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_matrix,
        display_labels=categories_notes
    )

    disp.plot(cmap='Greens', values_format='d')
    plt.title(f"Matrice de Confusion du Modèle Ordinal ({distr})")
    plt.show(block=False)
    plt.close()
    return model, result

def execute_random_forest(CHEMIN_DATABASE, CHEMIN_PERSONNES, CHEMIN_DATAPHOTO, max_depth=8, min_samples_leaf=5):
    X_train, X_test, y_train, y_test = preprocess_data_for_ordered_model(
        CHEMIN_DATABASE, CHEMIN_PERSONNES, CHEMIN_DATAPHOTO
    )

    model = RandomForestClassifier(n_estimators=1000, class_weight='balanced', random_state=42, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nRRandom Forest Classifier - Model Results")
    print(f"\nAccuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"\nClassification Report: {classification_report(y_test, y_pred, zero_division=0)}")
    conf_matrix = confusion_matrix(y_test, y_pred, labels=[1,2,3,4,5])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_matrix,
        display_labels=[1,2,3,4,5]
    )

    disp.plot(cmap='Greens', values_format='d')
    plt.title(f"Matrice de Confusion du RFC")
    plt.show(block=False)
    plt.close()

def preparer_donnees_sondage(filepath):
    """Charge et mappe les variables catégorielles du sondage."""
    df_sondage = pd.read_csv(filepath)

    # Mapping des catégories pour affichage lisible
    age_labels = [
        "moins de 18 ans", "18 à 24 ans", "25 à 34 ans", "35 à 49 ans", "50 à 64 ans",
        "65 à 69 ans", "70 à 74 ans", "75 à 85 ans", "plus de 85 ans"
    ]
    gender_labels = ["Femme", "Homme", "Autres", "Autres"]
    job_labels = [
        "Artisan·e·s, Commerçant·e·s, Agriculteurs", "Chef·fe·s d'entreprise",
        "Cadres, professions \nintellectuelles supérieures", "Professions intermédiaires",
        "Employé·e·s", "Ouvrier·e·s", "Étudiant·e·s", "Retraité·e·s", "Autres"
    ]
    bike_use_labels = [
        "Jamais", "Moins d'une \n fois par mois", "Une ou plusieurs \n fois par mois",
        "Une ou plusieurs \n fois par semaine", "Tous les jours ou presque"
    ]

    # Application des mappings
    df_sondage['age_cat'] = df_sondage['age'].map(
        lambda x: age_labels[int(x)] if pd.notnull(x) and int(x) < len(age_labels) else x)
    df_sondage['gender_cat'] = df_sondage['gender'].map(
        lambda x: gender_labels[int(x)] if pd.notnull(x) and int(x) < len(gender_labels) else x)
    df_sondage['job_cat'] = df_sondage['job'].map(
        lambda x: job_labels[int(x)] if pd.notnull(x) and int(x) < len(job_labels) else x)
    df_sondage['bike_use_cat'] = df_sondage['bike_use_frequency'].map(
        lambda x: bike_use_labels[int(x)] if pd.notnull(x) and int(x) < len(bike_use_labels) else x)

    # Catégorie lisible pour vélo électrique
    df_sondage['electric_bike_cat'] = df_sondage['electric_bike'].map(
        {True: 'Oui', False: 'Non', 'True': 'Oui', 'False': 'Non', 1: 'Oui', 0: 'Non'})

    return df_sondage

def plot_percentage_bar(data, col, title, ylabel, xlabel, save_path):
    """Génère et sauvegarde un graphique en barres horizontales de pourcentages."""
    labels = data[col].astype(str)
    counts = labels.value_counts(normalize=True).sort_values(ascending=True)
    percentages = counts.values * 100

    plt.figure(figsize=(12, 4))
    bars = plt.barh(counts.index, percentages, color=cmap.colors[:len(counts)], alpha=0.95)

    plt.yticks(rotation=45)
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter())

    # Suppression des bordures inutiles
    for spine in ['top', 'right', 'bottom']:
        plt.gca().spines[spine].set_visible(False)
    plt.gca().set_xticks([])

    # Ajout des labels sur les barres
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2, f'{percentages[i]:.1f}%', va='center',
                 fontsize=10)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(rf"{save_path}/survey/répartition_{col}.png")
    plt.close()

def plot_bike_types(df, save_path):
    """Génère et sauvegarde la répartition des types de vélos."""
    bike_type_labels = {
        'type_bike_road': 'Vélo de route',
        'type_bike_allroad': 'VTT et VTC',
        'type_bike_city': 'Vélo de ville'
    }

    bike_type_counts = {label: df[col].sum() for col, label in bike_type_labels.items() if col in df.columns}

    labels = list(bike_type_counts.keys())[::-1]
    counts = [bike_type_counts[label] for label in labels]
    total = sum(counts) if sum(counts) > 0 else 1
    percentages = [count / total * 100 for count in counts]

    plt.figure(figsize=(7, 4))
    bars = plt.barh(labels, percentages, color=cmap.colors[:len(labels)], alpha=0.95)
    plt.ylabel("Type de vélo")
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter())

    for spine in ['top', 'right', 'bottom']:
        plt.gca().spines[spine].set_visible(False)
    plt.gca().set_xticks([])
    plt.tight_layout()

    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2, f'{percentages[i]:.1f}%', va='center',
                 fontsize=10)

    plt.savefig(rf"{save_path}/survey/répartition_type_bike.png")
    plt.close()

def analyse_best_worst(filepath, save_path):
    """Effectue l'analyse Best-Worst Scaling et affiche les résultats."""
    df_sondage = pd.read_csv(filepath)
    df_sondage = df_sondage[df_sondage["Faites vous du vélo ?"].isin(["Jamais"])]

    plus_important_cols = [col for col in df_sondage.columns if "[Plus important]" in col]
    moins_important_cols = [col for col in df_sondage.columns if "[Moins important]" in col]

    all_items = pd.unique(df_sondage[plus_important_cols + moins_important_cols].values.ravel())
    all_items = [item for item in all_items if pd.notna(item)]

    results = pd.DataFrame(index=all_items)
    results["Plus Important"] = 0
    results["Moins Important"] = 0

    for item in all_items:
        best_count = (df_sondage[plus_important_cols] == item).sum().sum()
        worst_count = (df_sondage[moins_important_cols] == item).sum().sum()
        total_presented = len(df_sondage) * 6
        not_chosen = total_presented - best_count - worst_count

        results.loc[item, "Plus Important"] = best_count
        results.loc[item, "Moins Important"] = worst_count
        results.loc[item, "Pas choisi"] = not_chosen
        results.loc[item, "B-W Score"] = (best_count - worst_count) / total_presented

    results = results.sort_values("B-W Score", ascending=True)

    # Graphique empilé
    fig, ax = plt.subplots(figsize=(20, 20))
    bottom = None
    labels_bar = ["Plus Important", "Moins Important", "Pas choisi"]
    bars = []
    bar_colors = cmap.colors[1:4]

    for i, col in enumerate(labels_bar):
        bar = ax.barh(results.index, results[col], left=bottom, color=bar_colors[i], label=labels_bar[i])
        bars.append(bar)
        bottom = results[col] if bottom is None else bottom + results[col]

    # Scores
    for idx, score in enumerate(results["B-W Score"]):
        ax.text(bottom.iloc[idx] + 2, idx, f"{score:.2f}", va="center", fontsize=12)

    # Nombres sur colonnes
    for bar in bars:
        ax.bar_label(bar, fmt='%.0f', padding=3, label_type='center')
    ax.legend(loc='upper left', title="Choix", fontsize=10)
    for spine in ['top', 'right', 'bottom']:
        plt.gca().spines[spine].set_visible(False)
    plt.gca().set_xticks([])
    plt.yticks(rotation=45)
    plt.savefig(rf"{save_path}/survey/BW_score.png")
    plt.close()
    return results