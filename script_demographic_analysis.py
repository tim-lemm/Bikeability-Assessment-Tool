import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Chargement des données
# Assurez-vous que les fichiers CSV sont dans le même dossier que votre script
df_notes = pd.read_csv('data/survey/data-base.csv')
df_personnes = pd.read_csv('data/survey/data-personnes.csv')

# 2. Fusion des jeux de données
# On regroupe les notes et les profils grâce à 'id_personne'
df = pd.merge(df_notes, df_personnes, on='id_personne')

# Liste des variables catégorielles à analyser
variables_cibles = [
    'age',
    'job',
    'bike_use_frequency',
    'gender',
    'bike_ownership',
    'electric_bike'
]

# 3. Calcul des moyennes et écart-types
print("--- MOYENNES ET ÉCART-TYPES DES NOTES ---")
for var in variables_cibles:
    # On groupe par la variable et on calcule la moyenne et l'écart-type (std)
    stats = df.groupby(var)['note'].agg(['mean', 'std']).reset_index()
    # On renomme les colonnes pour plus de clarté
    stats.columns = [var, 'Moyenne', 'Écart-type']
    print(f"\nStatistiques en fonction de : {var}")
    print(stats.to_string(index=False))

# 4. Calcul des corrélations (Méthode de Pearson par défaut)
print("\n--- CORRÉLATIONS ---")
corr_age_note = df['age'].corr(df['note'])
corr_freq_note = df['bike_use_frequency'].corr(df['note'])

print(f"Corrélation entre l'âge et la note : {corr_age_note:.3f}")
print(f"Corrélation entre la fréquence d'utilisation et la note : {corr_freq_note:.3f}")

# ==========================================
# 5. VISUALISATIONS (Résumé complet)
# ==========================================
sns.set_theme(style="whitegrid")

# A. Graphiques de distribution (Boxplots) pour les catégories
fig, axes = plt.subplots(3, 2, figsize=(15, 16))
axes = axes.flatten()

for i, var in enumerate(variables_cibles):
    # Le boxplot est idéal car il montre la médiane, la dispersion (écart-type) et les valeurs aberrantes
    sns.boxplot(data=df, x=var, y='note', ax=axes[i], palette="Set2", hue=var, legend=False)
    axes[i].set_title(f'Distribution des notes par {var}', fontsize=12, fontweight='bold')
    axes[i].set_ylabel('Note')
    axes[i].set_xlabel(var)

plt.tight_layout()
plt.show()

# B. Graphiques de corrélation (Nuages de points avec droite de régression)
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

# Âge vs Note
sns.regplot(data=df, x='age', y='note', ax=axes2[0],
            scatter_kws={'alpha':0.1, 'color': 'blue'}, # Alpha faible pour mieux voir la densité
            line_kws={'color':'red', 'linewidth': 2}, x_jitter=0.1, y_jitter=0.1)
axes2[0].set_title(f"Tendance : Âge vs Note\n(Corrélation : {corr_age_note:.2f})", fontweight='bold')

# Fréquence d'utilisation vs Note
sns.regplot(data=df, x='bike_use_frequency', y='note', ax=axes2[1],
            scatter_kws={'alpha':0.1, 'color': 'green'},
            line_kws={'color':'red', 'linewidth': 2}, x_jitter=0.1, y_jitter=0.1)
axes2[1].set_title(f"Tendance : Fréquence d'utilisation vs Note\n(Corrélation : {corr_freq_note:.2f})", fontweight='bold')

plt.tight_layout()
plt.show()

# ==========================================
# C. Graphiques de comptage (Nombre d'individus uniques)
# ==========================================
# Création d'une nouvelle figure avec 6 sous-graphiques
fig3, axes3 = plt.subplots(3, 2, figsize=(15, 16))
axes3 = axes3.flatten()

for i, var in enumerate(variables_cibles):
    # 1. On compte les occurrences et on divise par 17 (nombre de photos par personne)
    comptes_ajustes = df[var].value_counts() / 17

    # 2. On utilise barplot au lieu de countplot pour fournir nos propres valeurs
    sns.barplot(
        x=comptes_ajustes.index,
        y=comptes_ajustes.values,
        ax=axes3[i],
        palette="viridis",
        hue=comptes_ajustes.index,
        legend=False
    )

    # 3. Mise en forme du graphique
    axes3[i].set_title(f'Nombre de personnes uniques par {var}', fontsize=12, fontweight='bold')
    axes3[i].set_ylabel('Nombre de personnes')
    axes3[i].set_xlabel(var)

    # 4. Ajout des valeurs exactes au-dessus de chaque barre
    for container in axes3[i].containers:
        # On convertit en entier (int) pour ne pas afficher de virgules (ex: 5.0 personnes)
        labels = [f'{int(v.get_height())}' for v in container]
        axes3[i].bar_label(container, labels=labels)

plt.tight_layout()
plt.show()
