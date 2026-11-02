import pandas as pd
import shap
import joblib
import os
import matplotlib.pyplot as plt
from pathlib import Path

from config import RANDOM_STATE
from utils_benchmark import load_and_prepare_data

base_csv = "data/survey/data-base.csv"
personnes_csv = "data/survey/data-personnes.csv"
photos_csv = "data/survey/data-photos.csv"
model_filepath = Path(f"outputs/photo_people/model_results/models/")
shap_filepath = Path(f"outputs/photo_people/model_results/shap_values/")
os.makedirs(shap_filepath, exist_ok=True)
list_model_name = [
    f.replace(".joblib", "")
    for f in os.listdir(model_filepath)
    if f.endswith(".joblib") and not f.startswith(".")
]
print(list_model_name)

X, y = load_and_prepare_data(base_csv, personnes_csv, photos_csv, people=True)
background_data = X.sample(1, random_state=RANDOM_STATE)

# for model_name in list_model_name:
#     print(f"Traitement de : {model_name}")
#     model = joblib.load(f"{model_filepath}/{model_name}.joblib")
#     explainer = shap.KernelExplainer(model.predict_proba, background_data)
#     shap_values = explainer.shap_values(X)
#     joblib.dump(shap_values, f"{shap_filepath}/shap_values_{model_name}.joblib")

shap_values = joblib.load(f"{shap_filepath}/shap_values_ada_boost.joblib")
for class_idx in range(shap_values.shape[2]):
    shap.plots.force(shap_values[:, :, class_idx], X)
    plt.show()