from stepmix.stepmix import StepMix
from config import *
from utils_latent_class import *

base_csv = "data/survey/data-base.csv"
personnes_csv = "data/survey/data-personnes.csv"
photos_csv = "data/survey/data-photos.csv"

df = lc_load_and_prepared_data(base_csv, personnes_csv, photos_csv)
df_personne = lc_transform_df_for_lc(df)
df_personne_enc = lc_encode_df_for_lc(df_personne)

list_n_classes = range(1, 9)
dict_variables = {
    "Only mean": ["note_mean"],
    "Only median": ["note_med"],
    "Only std": ["note_std"],
    "med, q25, q75": ["note_med", "note_q25", "note_q75"],
    "med, std" : ["note_med", "note_std"]
}

drop_cols = [c for c in (col_stat + col_note + ["id_personne"]) if c in df_personne_enc.columns]
X = df_personne_enc.drop(columns=drop_cols)

results = []

for n_classes in list_n_classes:
    for name, variables in dict_variables.items():
        y = df_personne_enc[variables]

        stepmix_model = StepMix(
            n_components=n_classes,
            n_steps=3,
            random_state=RANDOM_STATE,
            measurement="continuous",
            verbose=False,
        )
        stepmix_model.fit(X, y)

        results.append({
            "name": name,
            "n_class": n_classes,
            "aic": stepmix_model.aic(X, y),
            "bic": stepmix_model.bic(X, y),
        })

df_result_lc_gc = pd.DataFrame(results)


fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

metrics = [
    ("aic", "AIC selon le nombre de classes", axes[0]),
    ("bic", "BIC selon le nombre de classes", axes[1]),
]

for metric, title, ax in metrics:
  for name, grp in df_result_lc_gc.groupby("name", sort=False):
    grp = grp.sort_values("n_class")
    (line,) = ax.plot(
        grp["n_class"], grp[metric], marker="o", linewidth=1, label=name
    )
    # Marquer le minimum pour ce jeu de variables
    best_row = grp.loc[grp[metric].idxmin()]
    ax.scatter(
        best_row["n_class"],
        best_row[metric],
        color=line.get_color(),
        s=90,
        zorder=5,
        marker="*",
    )

  ax.set_title(title)
  ax.set_ylabel(metric.upper())
  ax.set_xticks(sorted(df_result_lc_gc["n_class"].unique()))
  ax.grid(True, linestyle="--", alpha=0.4)
  ax.legend(title="Variables", bbox_to_anchor=(1.02, 1), loc="upper left")

axes[0].set_xlabel("Nombre de classes")
axes[1].set_xlabel("Nombre de classes")

plt.tight_layout()
plt.show()

best_aic = df_result_lc_gc.loc[
    df_result_lc_gc.groupby("name")["aic"].idxmin(),
    ["name", "n_class", "aic"],
].rename(columns={"n_class": "best_n_class_aic"})
best_bic = df_result_lc_gc.loc[
    df_result_lc_gc.groupby("name")["bic"].idxmin(),
    ["name", "n_class", "bic"],
].rename(columns={"n_class": "best_n_class_bic"})

summary_best = best_aic.merge(best_bic, on="name")
print(summary_best.to_string(index=False))