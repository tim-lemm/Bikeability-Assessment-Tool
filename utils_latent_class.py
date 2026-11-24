from utils_benchmark import import_and_merge_data_base
from sklearn.preprocessing import LabelEncoder

categorical_cols = ["nbr_lane", "speed", "slope", "type", "green", "age", "gender", "job", "bike_use_frequency"]
covariate_cols = ["gender", "job", "age", "bike_use_frequency", "electric_bike", "bike_ownership", "type_bike"]
col_note = [f"note_{i}" for i in range(1, 18)]
col_stat = ["note_mean", "note_std", "note_min", "note_max", "note_med", "note_q25", "note_q75"]

def lc_load_and_prepared_data(base_csv, personnes_csv, photos_csv):
    df = import_and_merge_data_base(base_csv, personnes_csv, photos_csv)
    df.drop(["postal_code", "type_bike_fold", "type_bike_cargo", "type_bike_recumbent", "traffic"], axis=1,
            inplace=True)
    df["electric_bike"] = df["electric_bike"].astype(int)

    df["type_bike"] = "city"
    df.loc[df["type_bike_allroad"] == 1, "type_bike"] = "allroad"
    df.loc[df["type_bike_road"] == 1, "type_bike"] = "road"

    df.drop(["type_bike_allroad", "type_bike_city", "type_bike_road"], axis=1, inplace=True)
    return df

def lc_transform_df_for_lc(df):
    df_personne = df.copy()
    df_personne.drop(["nbr_lane", "speed", "slope", "green", "type"], axis=1, inplace=True)
    cols_personne = [c for c in df_personne.columns if c not in ["id_photo", "id_personne", "note"]]
    df_notes = df_personne.pivot(index="id_personne", columns="id_photo", values="note")
    df_notes.columns = [f"note_{col}" for col in df_notes.columns]
    df_notes.reset_index(inplace=True)

    df_carac = df_personne[["id_personne"] + cols_personne].drop_duplicates(subset="id_personne")

    df_personne = df_carac.merge(df_notes, on="id_personne", how="left")

    col_note = [f"note_{i}" for i in range(1, 18)]
    df_personne["note_mean"] = df_personne[col_note].mean(axis=1)
    df_personne["note_std"] = df_personne[col_note].std(axis=1)
    df_personne["note_min"] = df_personne[col_note].min(axis=1)
    df_personne["note_max"] = df_personne[col_note].max(axis=1)
    df_personne["note_med"] = df_personne[col_note].median(axis=1)
    quantiles = df_personne[col_note].quantile([0.25, 0.75], axis=1)
    df_personne["note_q25"] = quantiles.loc[0.25]
    df_personne["note_q75"] = quantiles.loc[0.75]

    return df_personne

def lc_encode_df_for_lc (df):
    original_mappings = {}
    label_encoder = LabelEncoder()
    for col in covariate_cols:
        df[col] = label_encoder.fit_transform(df[col])
        original_mappings[col] = {i: label for i, label in enumerate(label_encoder.classes_)}
    return df

def plot_notes_by_class(df: pd.DataFrame, note_id: int | None = None):
    """Affiche un histogramme (barres groupées en %) des notes selon 'class'.

    - note_id=None : agrège toutes les colonnes note_1 .. note_17.
    - note_id=1..17 : cible uniquement la colonne note_{note_id}.
    """
    note_cols = [f"note_{i}" for i in range(1, 18)]

    if note_id is not None:
        col = f"note_{note_id}"
        if col not in df.columns:
            raise ValueError(f"Colonne '{col}' introuvable dans le DataFrame.")
        sub = df[["class", col]].dropna().rename(columns={col: "note"})
        title = f"Répartition de {col} selon class (%)"
    else:
        sub = (
            df[["class"] + note_cols]
            .melt(id_vars="class", value_vars=note_cols, value_name="note")
            .dropna(subset=["class", "note"])[["class", "note"]]
        )
        title = "Répartition de toutes les notes (note_1 à note_17) selon class"

    # Calcul des pourcentages à l'intérieur de chaque 'class' (somme = 100 % par classe)
    ct = (
        pd.crosstab(sub["note"], sub["class"], normalize="columns")
        * 100
    )

    # S'assurer que les notes (ex. 1 à 5) sont triées dans l'ordre croissant
    notes_sorted = sorted(ct.index.unique())
    ct = ct.reindex(notes_sorted, fill_value=0.0)

    classes = sorted(ct.columns.unique())
    n_notes = len(notes_sorted)
    n_classes = len(classes)

    x = np.arange(n_notes)
    width = 0.8 / max(n_classes, 1)

    fig, ax = plt.subplots(figsize=(9, 5))

    for idx, cls in enumerate(classes):
        offsets = (idx - (n_classes - 1) / 2) * width
        heights = ct[cls].values
        bars = ax.bar(x + offsets, heights, width=width, label=f"class {cls}")

        # Étiquettes de pourcentage au-dessus de chaque barre
        for bar, h in zip(bars, heights):
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.8,
                    f"{h:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_title(title)
    ax.set_xlabel("Note")
    ax.set_ylabel("Pourcentage par classe (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in notes_sorted])
    ax.set_ylim(0, max(ct.values.max() * 1.15, 10))
    ax.legend(title="Classe")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()