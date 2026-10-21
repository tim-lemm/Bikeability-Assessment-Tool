import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from utils_model_handeling import apply_bi_model
from pathlib import Path
import os

mapping_infra = {
    "Pas d'aménagement cyclable": 0,
    "Chaussée partagée aménagée": 1,
    "Aménagement séparé partagé": 2,
    "Aménagement séparé non partagé": 3
}

gdf_routes = gpd.read_file("data/gdf_routes.gpkg")
gdf_routes["type"] = gdf_routes["nom_amenagement"].map(mapping_infra)
gdf_routes["speed"] = gdf_routes["vitesse_moyenne_vl"]
gdf_routes["green"] = gdf_routes["green_overlap_percent"]
gdf_routes["nbr_lane"] = gdf_routes["nombre_de_voies"]
gdf_routes["slope"] = gdf_routes["pente_2"]

gdf_routes_clean = gdf_routes.dropna(subset=["type", "speed", "green", "nbr_lane", "slope"])

model_filepath = Path("outputs/model_results/models/")
list_model_name = [
    f.replace(".joblib", "")
    for f in os.listdir(model_filepath)
    if f.endswith(".joblib") and not f.startswith(".")
]

dict_notes = {}
for model_name in list_model_name:
    model = joblib.load(f"{model_filepath}/{model_name}.joblib")
    gdf_routes_with_notes = apply_bi_model(gdf_routes_clean, model)
    dict_notes[model_name] = gdf_routes_with_notes['note']

all_notes = [1, 2, 3, 4, 5]

cmap = plt.get_cmap('Spectral')
color_mapping = {note_val: cmap(i / max(1, len(all_notes) - 1)) for i, note_val in enumerate(all_notes)}

Path("outputs/maps").mkdir(parents=True, exist_ok=True)

for model_name in list_model_name:
    gdf_routes_with_notes = gdf_routes_clean.copy()
    gdf_routes_with_notes['note'] = pd.Categorical(dict_notes[model_name], categories=all_notes)
    gdf_routes_with_notes['length'] = gdf_routes_with_notes.geometry.length

    fig, (ax_map, ax_hist) = plt.subplots(2, 1, figsize=(8, 11), gridspec_kw={'height_ratios': [3, 1]})
    fig.set_dpi(600)
    fig.suptitle(model_name)

    gdf_routes_with_notes.plot(column="note", legend=True, categorical=True, cmap="Spectral", ax=ax_map)
    ax_map.set_axis_off()

    note_counts = gdf_routes_with_notes['note'].value_counts().reindex(all_notes, fill_value=0)
    note_lengths = gdf_routes_with_notes.groupby('note', observed=False)['length'].sum().reindex(all_notes, fill_value=0)

    x = np.arange(len(all_notes))
    width = 0.35
    colors_list = [color_mapping[n] for n in all_notes]

    bars_count = ax_hist.bar(x - width/2, note_counts.values, width, label='Count', color=colors_list, edgecolor='black', linewidth=0.5, alpha=0.9)
    ax_hist.bar_label(bars_count, padding=3, fmt='%d')
    ax_hist.set_ylabel("Count", color='black')

    ax_hist2 = ax_hist.twinx()
    bars_length = ax_hist2.bar(x + width/2, note_lengths.values, width, label='Length', color=colors_list, edgecolor='black', linewidth=0.5, alpha=0.5)
    ax_hist2.bar_label(bars_length, padding=3, fmt='%.1f')
    ax_hist2.set_ylabel("Total Length", color='gray')

    ax_hist.set_xticks(x)
    ax_hist.set_xticklabels([str(n) for n in all_notes])
    ax_hist.set_xlabel("Note")
    ax_hist.grid(axis='y', linestyle='--', alpha=0.7)

    ax_hist.spines['top'].set_visible(False)
    ax_hist2.spines['top'].set_visible(False)

    plt.tight_layout()
    plt.show()
    # plt.savefig(f"outputs/maps/map_{model_name}.png", dpi=600)
    # plt.close(fig)

    m = gdf_routes_with_notes.explore(column="note")
    m.save(f"outputs/maps/map_{model_name}.html")

gdf_routes_all_notes = gdf_routes_clean.copy()
for model_name, notes in dict_notes.items():
    gdf_routes_all_notes[f"note_{model_name}"] = notes

gdf_routes_all_notes.to_csv("outputs/maps/gdf_routes_all_notes.csv")

gdf_routes_comparaison = gdf_routes_all_notes.copy()
gdf_routes_comparaison['GB-ET'] = gdf_routes_comparaison["note_gradient_boost"] - gdf_routes_comparaison["note_extratrees"]
gdf_routes_comparaison['length'] = gdf_routes_comparaison.geometry.length



# Comparaison maps
fig, (ax_map, ax_hist) = plt.subplots(2, 1, figsize=(8, 11), gridspec_kw={'height_ratios': [3, 1]})
fig.set_dpi(600)
fig.suptitle("Comparaison GB - ET")

diff_notes = sorted(gdf_routes_comparaison['GB-ET'].dropna().unique())
gdf_routes_comparaison['GB-ET'] = pd.Categorical(gdf_routes_comparaison['GB-ET'], categories=diff_notes)

gdf_routes_comparaison.plot(column="GB-ET", categorical=True, legend=True, cmap="coolwarm", ax=ax_map)
ax_map.set_axis_off()

diff_counts = gdf_routes_comparaison['GB-ET'].value_counts().reindex(diff_notes, fill_value=0)
diff_lengths = gdf_routes_comparaison.groupby('GB-ET', observed=False)['length'].sum().reindex(diff_notes, fill_value=0)

x_diff = np.arange(len(diff_notes))
cmap_diff = plt.get_cmap('coolwarm')
color_list_diff = [cmap_diff(i / max(1, len(diff_notes) - 1)) for i in range(len(diff_notes))]

bars_count_diff = ax_hist.bar(x_diff - width/2, diff_counts.values, width, label='Count', color=color_list_diff, edgecolor='black', linewidth=0.5, alpha=0.9)
ax_hist.bar_label(bars_count_diff, padding=3, fmt='%d')
ax_hist.set_ylabel("Count", color='black')

ax_hist2 = ax_hist.twinx()
bars_length_diff = ax_hist2.bar(x_diff + width/2, diff_lengths.values, width, label='Length', color=color_list_diff, edgecolor='black', linewidth=0.5, alpha=0.5)
ax_hist2.bar_label(bars_length_diff, padding=3, fmt='%.1f')
ax_hist2.set_ylabel("Total Length", color='gray')

ax_hist.set_xticks(x_diff)
ax_hist.set_xticklabels([str(n) for n in diff_notes])
ax_hist.set_xlabel("Difference (GB - ET)")
ax_hist.grid(axis='y', linestyle='--', alpha=0.7)

ax_hist.spines['top'].set_visible(False)
ax_hist2.spines['top'].set_visible(False)

plt.tight_layout()
plt.savefig(f"outputs/maps/map_GB-ET.png", dpi=600)
plt.close(fig)