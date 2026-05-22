import pandas as pd
from utils_survey_analysis import (
    executer_regression_mean,
    executer_regression_all,
    executer_regression_tree_mean,
    executer_regression_tree_all,
    preparer_donnees_sondage,
    plot_percentage_bar,
    plot_bike_types,
    analyse_best_worst
)


def main():
    # Définition des chemins (avec 'r' pour éviter les avertissements d'échappement invalide)
    CHEMIN_SUMMARY = r"data/survey/summary.csv"
    CHEMIN_DATABASE = r"data/survey/data-base.csv"
    CHEMIN_PERSONNES = r"data/survey/data-personnes.csv"
    CHEMIN_DATAPHOTO = r"data/survey/data-photos.csv"
    CHEMIN_BEST_WORST = r"data/survey/data-all.csv"
    DOSSIER_IMAGES = r"outputs/"


    print("\n=== 1. Nettoyage et Préparation des Données du Sondage ===")
    try:
        df_sondage = preparer_donnees_sondage(CHEMIN_PERSONNES)
        print("\nDonnées du sondage chargées.")
    except FileNotFoundError:
        raise ValueError(f"Attention: Le fichier {CHEMIN_PERSONNES} n'a pas été trouvé.")


    print("\n=== 2. Génération des Graphiques de Répartition ===")
    plot_percentage_bar(df_sondage, 'gender_cat', "Répartition par genre (%)", "Genre", "Pourcentage", DOSSIER_IMAGES)
    plot_percentage_bar(df_sondage, 'age_cat', "Répartition par âge (%)", "Âge", "Pourcentage", DOSSIER_IMAGES)
    plot_percentage_bar(df_sondage, 'job_cat', "Répartition par occupation (%)", "Occupation", "Pourcentage", DOSSIER_IMAGES)
    plot_percentage_bar(df_sondage, 'bike_use_cat', "Répartition par fréquence d'utilisation du vélo (%)", "Fréquence", "Pourcentage", DOSSIER_IMAGES)
    plot_percentage_bar(df_sondage, 'electric_bike_cat', "Répartition par utilisation d'un vélo électrique (%)", "Possession d'un vélo électrique", "Pourcentage", DOSSIER_IMAGES)
    plot_bike_types(df_sondage, DOSSIER_IMAGES)

    print("\nImages de synthétisation créées.")


    print("\n=== 3. Analyse Best-Worst Scaling ===")
    try:
        analyse_best_worst(CHEMIN_BEST_WORST, DOSSIER_IMAGES)
        print("\nScores Best-Worst générés avec succès.")
    except FileNotFoundError:
        raise ValueError(f"Attention: Le fichier {CHEMIN_BEST_WORST} n'a pas été trouvé.")


    print("\n=== 4. Régressions linéaires ===")
    try:
        executer_regression_mean(CHEMIN_SUMMARY, test_size = 0.2, random_state = 42)
        executer_regression_all(CHEMIN_DATABASE, CHEMIN_PERSONNES, CHEMIN_DATAPHOTO, test_size = 0.2, random_state = 42)
        print("\nRégressions linéaires terminées.")
    except FileNotFoundError:
        raise ValueError(f"Attention: Le fichier {CHEMIN_SUMMARY} n'a pas été trouvé.")

    print("\n=== 5. Arbres ===")
    executer_regression_tree_mean(CHEMIN_SUMMARY, DOSSIER_IMAGES, test_size = 0.2, random_state = 42)
    executer_regression_tree_all(CHEMIN_DATABASE, CHEMIN_PERSONNES, CHEMIN_DATAPHOTO, DOSSIER_IMAGES, test_size = 0.2, random_state = 42, depth = None)

if __name__ == "__main__":
    main()