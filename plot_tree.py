import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# 1. Charger le modèle sauvegardé (remplacez par votre chemin de fichier)
model = joblib.load('outputs/model_results/models/classification_tree_optimized.joblib')

# 2. Créer la figure Matplotlib (ajustez la taille selon la complexité de l'arbre)
plt.figure(figsize=(15, 10), dpi=300)

# 3. Tracer l'arbre
plot_tree(
    model,
    feature_names=  [f"nbr_lane_{i}" for i in range(4)] + [f"speed_{i}" for i in range(4)] + \
                [f"slope_{i}" for i in range(3)] + [f"green_{i}" for i in range(3)] + \
                [f"type_{i}" for i in range(4)],
    filled=True,  # Colore les nœuds selon la pureté/classe dominante
    rounded=True,  # Arrondit les coins des boîtes
    precision=2,
    fontsize=7
)

# 4. Afficher le graphique
plt.show()