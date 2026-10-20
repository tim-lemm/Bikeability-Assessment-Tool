from utils_survey_analysis import (
    execute_regression_mean,
    execute_regression_all,
    execute_regression_tree_mean,
    execute_regression_tree_all,
    execute_random_forest,
    execute_ordered_model,
    preparer_donnees_sondage,
    plot_percentage_bar,
    plot_bike_types,
    analyse_best_worst
)
from utils_benchmark import *
import matplotlib.pyplot as plt

def main():
    CHEMIN_SUMMARY = r"data/survey/summary.csv"
    CHEMIN_DATABASE = r"data/survey/data-base.csv"
    CHEMIN_PERSONNES = r"data/survey/data-personnes.csv"
    CHEMIN_DATAPHOTO = r"data/survey/data-photos.csv"
    CHEMIN_BEST_WORST = r"data/survey/data-all.csv"
    DOSSIER_IMAGES = r"outputs/survey/"


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


    # print("\n=== 3. Analyse Best-Worst Scaling ===")
    # try:
    #     analyse_best_worst(CHEMIN_BEST_WORST, DOSSIER_IMAGES)
    #     print("\nScores Best-Worst générés avec succès.")
    # except FileNotFoundError:
    #     raise ValueError(f"Attention: Le fichier {CHEMIN_BEST_WORST} n'a pas été trouvé.")


def main2():
    base_path = "data/survey/data-base.csv"
    personnes_path = "data/survey/data-personnes.csv"
    photos_path = "data/survey/data-photos.csv"

    df_all = import_and_merge_data_base(base_path, personnes_path, photos_path)
    df_all = df_all[df_all['job'] == 6]
    df = df_all[df_all['postal_code'].between(69000, 70000, inclusive='neither')]

    # Compter le nombre de lignes par note
    note_counts = df['note'].value_counts().sort_index()

    fig, ax_hist = plt.subplots()
    cmap = plt.get_cmap('Spectral')
    num_categories = len(note_counts)

    colors = [cmap(i / max(1, num_categories - 1)) for i in range(num_categories)]

    bars = ax_hist.bar(
        note_counts.index.astype(str),
        note_counts.values,
        color=colors,
        edgecolor='black',
        linewidth=0.5
    )
    ax_hist.bar_label(bars, padding=3, fmt='%d')  # '%d' pour afficher un entier

    ax_hist.grid(axis='y', linestyle='--', alpha=0.7)
    ax_hist.set_xlabel("Note")
    ax_hist.set_ylabel("Nombre")

    ax_hist.spines['top'].set_visible(False)
    ax_hist.spines['right'].set_visible(False)

    plt.title("Distribution of given notes")
    # plt.show()
    plt.savefig("outputs/survey/ditrib_notes.png", dpi=300)
if __name__ == "__main__":
    main2()