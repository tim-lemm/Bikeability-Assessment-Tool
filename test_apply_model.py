import geopandas as gpd
from matplotlib import pyplot as plt
from pytensor.tensor.random import categorical
from utils_model_handeling import apply_bi_model
from pathlib import Path

mapping_infra = {
    "Pas d'aménagement cyclable":0,
    "Chaussée partagée aménagée":1,
    "Aménagement séparé partagé":2,
    "Aménagement séparé non partagé":3
}
 #preproccesing
gdf_routes = gpd.read_file("data/gdf_routes.gpkg")
gdf_routes["type"] = gdf_routes["nom_amenagement"].map(mapping_infra)
gdf_routes["speed"] = gdf_routes["vitesse_moyenne_vl"]
gdf_routes["green"] = gdf_routes["green_overlap_percent"]
gdf_routes["nbr_lane"] = gdf_routes["nombre_de_voies"]
gdf_routes["slope"] = gdf_routes["pente_2"]

# gdf_routes["green_overlap_percent"].hist(bins=50, edgecolor='black', linewidth=0.5)
# plt.title("green_overlap_percent distribution")
# plt.show()
# gdf_routes["pente_1"].hist(bins=50, edgecolor='black', linewidth=0.5)
# plt.title("pente_1 distribution")
# plt.show()
# gdf_routes["vitesse_moyenne_vl"].hist(bins=50, edgecolor='black', linewidth=0.5)
# plt.title("vitesse_moyenne_vl distribution")
# plt.show()
# gdf_routes["nombre_de_voies"].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
# plt.show()
# gdf_routes["nom_amenagement"].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
# plt.show()


gdf_routes_clean = gdf_routes.dropna(subset = ["type","speed","green","nbr_lane","slope"])

#model application
model_filepath = Path("data/models")
list_model_name = [f.name for f in model_filepath.iterdir() if f.is_file()]
list_model_name = [Path(f).stem for f in list_model_name]
dict_notes={}

for model_name in list_model_name:
    gdf_routes_with_notes = apply_bi_model(gdf_routes_clean, model_name)
    dict_notes[model_name] = gdf_routes_with_notes['note']

    fig, (ax_map, ax_hist) = plt.subplots(2, 1, figsize=(7, 10), gridspec_kw={'height_ratios': [3, 1]})
    fig.set_dpi(600)
    fig.suptitle(model_name)

    gdf_routes_with_notes.plot(column="note", legend=True, categorical=True, cmap="Spectral", ax=ax_map)
    ax_map.set_axis_off()

    gdf_routes_with_notes['length'] = gdf_routes_with_notes.geometry.length
    note_lengths = gdf_routes_with_notes.groupby('note')['length'].sum().sort_index()

    cmap = plt.get_cmap('Spectral')
    num_categories = len(note_lengths)
    colors = [cmap(i / max(1, num_categories - 1)) for i in range(num_categories)]

    bars = ax_hist.bar(note_lengths.index.astype(str), note_lengths.values, color=colors, edgecolor='black',
                       linewidth=0.5)
    ax_hist.bar_label(bars, padding=3, fmt='%.1f')

    ax_hist.grid(axis='y', linestyle='--', alpha=0.7)

    ax_hist.set_xlabel("Note")
    ax_hist.set_ylabel("Total Length")

    ax_hist.spines['top'].set_visible(False)
    ax_hist.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"outputs/maps/map_{model_name}.png", dpi=600)
    # plt.show()

    # save html map
    m = gdf_routes_with_notes.explore(column="note")
    m.save(f"outputs/maps/map_{model_name}.html")

# save all results
gdf_routes_all_notes = gdf_routes_clean.copy()
for model_name, notes in dict_notes.items():
    gdf_routes_all_notes[f"note_{model_name}"] = notes

gdf_routes_all_notes.to_csv("outputs/maps/gdf_routes_all_notes.csv")

# gdf_routes_all_notes = gpd.read_file("outputs/maps/gdf_routes_all_notes.csv")
# plot maps comparaison Ada - RAF
note_columns = ['note_ada_boost_optimized', "note_classification_tree_optimized", "note_extratrees_optimized",
                                                 "note_gradient_boost_optimized", "note_histgradient_boost_optimized", "note_random_forest_optimized"]
gdf_routes_comparaison = gdf_routes_all_notes.copy()
gdf_routes_comparaison = gdf_routes_comparaison[['geometry', 'cleabs',"type","speed","green","nbr_lane","slope",
                                                 'nom_collaboratif_gauche', 'nom_collaboratif_droite',
                                                 ] + note_columns]


gdf_routes_comparaison[note_columns]=gdf_routes_comparaison[note_columns].astype(int)

gdf_routes_comparaison['ada-raf']=gdf_routes_comparaison["note_ada_boost_optimized"]-gdf_routes_comparaison["note_random_forest_optimized"]

fig, (ax_map, ax_hist) = plt.subplots(2, 1, figsize=(7, 10), gridspec_kw={'height_ratios': [3, 1]})
fig.set_dpi(600)
fig.suptitle("Comparaison Ada - RAF")

gdf_routes_comparaison.plot(column="ada-raf",categorical=True, legend=True, cmap="Spectral", ax=ax_map)
ax_map.set_axis_off()

note_counts = gdf_routes_comparaison['ada-raf'].value_counts().sort_index()

cmap = plt.get_cmap('Spectral')
num_categories = len(note_counts)
colors = [cmap(i / max(1, num_categories - 1)) for i in range(num_categories)]

bars = ax_hist.bar(note_counts.index.astype(str), note_counts.values, color=colors, edgecolor='black',
                       linewidth=0.5)
ax_hist.bar_label(bars, padding=3)

ax_hist.grid(axis='y', linestyle='--', alpha=0.7)

ax_hist.set_xlabel("Note")
ax_hist.set_ylabel("Count")

ax_hist.spines['top'].set_visible(False)
ax_hist.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f"outputs/maps/map_ada-raf.png", dpi=600)
# plt.show()