import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_histo(df, width = 20):
    colors = np.where(df["mean"] >= 0, "#1f77b4", "#d62728")
    fig, ax = plt.subplots(figsize=(width, 6))
    ax.set_axisbelow(True)
    ax.bar(
        df["Unnamed: 0"], df["mean"], color=colors
    )
    plt.xticks(
        rotation=45, ha="right"
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.grid(axis="x", linestyle="--", alpha=0.7, zorder=1)
    ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=1)
    plt.tight_layout()
    plt.show()

def plot_kde(df_1, df_2):
    sns.kdeplot(
        df_1, x="mean", fill=True, color="teal", alpha=0.5, label="personnes", )
    sns.kdeplot(
        df_2, x="mean", fill=True, color="orange", alpha=0.5, label="photo", )
    plt.legend()
    plt.title("Distribution mean values of personnes et photo distribution")
    plt.tight_layout()
    plt.xlabel("mean")
    plt.ylabel("density")
    plt.grid(True)
    plt.show()

results_variables = pd.read_excel("resultats_variables.xlsx")

results_ordered = results_variables.sort_values(by="mean")
results_beta = results_ordered[~results_ordered["Unnamed: 0"].str.startswith(('u_', 'v_'), na=False)]
plot_histo(results_beta, width = 20)
results_u_personnes = results_ordered[results_ordered["Unnamed: 0"].str.startswith(('u_'), na=False)]
results_v_photo = results_ordered[results_ordered["Unnamed: 0"].str.startswith(('v_'), na=False)]
plot_kde(results_u_personnes, results_v_photo)
plot_histo(results_u_personnes, width = 40)
plot_histo(results_v_photo, width = 20)



results_cutpoints = pd.read_excel("seuils.xlsx",index_col=0)
df_cutpoints = results_cutpoints[results_cutpoints.index.str.contains("cutpoints", na=False)]
cutpoints_means = sorted(df_cutpoints["mean"].values)
sns.set_theme(style="white")

# Définition des limites dynamiques de l'axe X selon vos vraies valeurs
min_x = cutpoints_means[0] - 1.5
max_x = cutpoints_means[-1] + 1.5

fig, ax = plt.subplots(figsize=(12, 4))

# Couleurs pastel élégantes pour les 5 zones de notes
colors = ["#ffadad", "#ffd166", "#e9ff70", "#a0c4ff", "#bdb2ff"]
notes_labels = [
    "Note 1",
    "Note 2",
    "Note 3",
    "Note 4",
    "Note 5",
]

# Délimitation des blocs de couleur
edges = [min_x] + cutpoints_means + [max_x]

# Remplissage des zones de couleur et ajout du texte descriptif
for i in range(5):
    ax.axvspan(edges[i], edges[i + 1], alpha=0.6)
    center_x = (edges[i] + edges[i + 1]) / 2
    ax.text(
        center_x,
        0.5,
        notes_labels[i],
        ha="center",
        va="center",
        weight="bold",
        fontsize=11,
        color="#2f3e46",
    )

# Ajout des lignes pointillées noires pour matérialiser les seuils
for idx, cp in enumerate(cutpoints_means):
    ax.axvline(x=cp, color="#212529", linestyle="--", linewidth=1.8)
    # Affichage de la valeur mathématique au-dessus de chaque ligne
    ax.text(
        cp,
        0.98,
        f"$\\gamma_{idx+1} = {cp:.2f}$",
        ha="center",
        va="bottom",
        weight="bold",
        fontsize=11,
        bbox=dict(facecolor="white", alpha=0.9, boxstyle="round,pad=0.3"),
    )

ax.set_xlim(min_x, max_x)
ax.set_ylim(0, 1.15)

ax.get_yaxis().set_visible(False)

sns.despine(left=True, right=True, top=True)

ax.set_xlabel(
    "$\\mu$", labelpad=12, fontsize=12
)
ax.set_title(
    "Cutpoints",
    pad=25,
    fontsize=14,
    weight="bold",
)

plt.tight_layout()
plt.show()