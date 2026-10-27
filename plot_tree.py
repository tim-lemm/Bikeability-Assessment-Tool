import joblib
from sklearn.tree import export_graphviz
import graphviz

# 1. Charger le modèle sauvegardé (remplacez par votre chemin de fichier)
model = joblib.load("outputs/photo_people/model_results/models/classification_tree.joblib")



# Générer les données au format dot
dot_data = export_graphviz(
    model,
    out_file=None,
    feature_names=model.feature_names_in_,
    filled=True,
    rounded=True,
    special_characters=True,
    precision=2
)

dot_data = dot_data.replace("&le; 0.5", "= 0")

# Afficher ou sauvegarder l'arbre proprement
graph = graphviz.Source(dot_data)
graph.render("outputs/photo_people/model_results/arbre_decision", view=True, format="png")