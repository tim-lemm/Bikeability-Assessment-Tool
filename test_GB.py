import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, LeaveOneOut
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, classification_report, confusion_matrix, ConfusionMatrixDisplay
import shap
import lightgbm as lgb
from utils_MCLR import load_and_preprocess_data
from joblib import Parallel, delayed

## === Importation des données ===
print("=== Importation des données ===")

base_csv = "data/survey/data-base.csv"
personnes_csv = "data/survey/data-personnes.csv"
photos_csv = "data/survey/data-photos.csv"

# 1. Chargement des données
df, dims, coords = load_and_preprocess_data(base_csv, personnes_csv, photos_csv)

liste_cat = ["nbr_lane", "speed", "slope", "type", "green"]
X = df[["nbr_lane", "speed", "slope", "type", "green"]]

for cat in liste_cat:
    X[cat] = X[cat].astype("category")

y = df["note"]

# 2. Division Train / Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Taille du jeu d'entraînement : {X_train.shape}")
print(f"Taille du jeu de test : {X_test.shape}")

## === Modèle ===
print("=== Entrainement du modèle ===")

# 1. Initialisation du modèle

model = HistGradientBoostingClassifier()
# 2. Entraînement
model.fit(X_train, y_train)

# 3. Évaluation rapide
y_pred_continuous = model.predict(X_test)
y_pred_discrete = np.clip(np.round(y_pred_continuous), 1, 5).astype(int)

mae = mean_absolute_error(y_test, y_pred_continuous)

print(f"--- Performances du modèle ---")
print(f"Erreur moyenne (MAE) : {mae:.2f}")
print("\nRapport de classification (après arrondi) :")
print(classification_report(y_test, y_pred_discrete, zero_division=0))

## === LOO CV ===

loo = LeaveOneOut()
n_samples = len(X)

y_true_all = []
y_pred_all = []


def run_single_fold(train_idx, test_idx):
    X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
    y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

    model_fold = HistGradientBoostingClassifier()
    model_fold.fit(X_train_fold, y_train_fold)

    return y_test_fold.values[0], model_fold.predict(X_test_fold)[0]

results = Parallel(n_jobs=-1)(
    delayed(run_single_fold)(train_idx, test_idx) for train_idx, test_idx in loo.split(X)
)

y_true_all, y_pred_all = zip(*results)

y_true_all = np.array(y_true_all)
y_pred_all = np.array(y_pred_all)
y_true_all = np.array(y_true_all).ravel()
y_pred_all = np.array(y_pred_all).ravel()

mae_global = mean_absolute_error(y_true_all, y_pred_all)
r2_global = r2_score(y_true_all, y_pred_all)

y_pred_rounded = np.clip(np.round(y_pred_all), 1, 5).astype(int)

labels = [1, 2, 3, 4, 5]
cm = confusion_matrix(y_true_all, y_pred_rounded, labels=labels)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=labels,
    yticklabels=labels
)

plt.title("Matrice de Confusion (LOOCV)", fontsize=14, pad=15)
plt.ylabel("Notes Réelles", fontsize=12)
plt.xlabel("Notes Prédites", fontsize=12)
plt.tight_layout()
plt.show()

print("\n--- RÉSULTATS GLOBAUX LOOCV ---")
print(f"MAE Globale : {mae_global:.3f}")
print(f"R² Global   : {r2_global:.3f}")

df_res = pd.DataFrame({'Réel': y_true_all, 'Prédit (Brut)': y_pred_all, 'Prédit (Arrondi)': y_pred_rounded})
print("\nAperçu des premières prédictions :")
print(df_res.head(10))

## === SHAP ===
print("=== Analyse SHAP ===")

# 1. Sélection d'un échantillon pour l'interprétation
X_explain = X_test.sample(500, random_state=42)

# 2. Création de l'explainer (spécifique aux modèles d'arbres)
explainer = shap.TreeExplainer(model)

# 3. Calcul des SHAP values
shap_values = explainer(X_explain)

## === VIZ ===
print("=== Visualization ===")

shap.summary_plot(shap_values, X_explain)