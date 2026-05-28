import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.causal_discovery import HillClimbSearch
from pgmpy.parameter_estimator import DiscreteMLE, DiscreteBayesianEstimator
from pgmpy.inference import VariableElimination
import networkx as nx
import matplotlib.pyplot as plt


# https://github.com/pgmpy/pgmpy_tutorials/blob/master/notebooks/11.%20A%20Bayesian%20Network%20to%20model%20the%20influence%20of%20energy%20consumption%20on%20greenhouse%20gases%20in%20Italy.ipynb

np.random.seed(42)
n_samples = 1500

infra = np.random.choice(['piste_separee', 'bande_peinte', 'rien'], size=n_samples)
vitesse = np.random.choice(['30', '50', '80'], size=n_samples)
pente = np.random.choice(['faible', 'forte'], size=n_samples)
voies = np.random.choice(['1_voie', 'multi_voies'], size=n_samples)
espace_vert = np.random.choice(['oui', 'non'], size=n_samples)
notes = []

for i in range(n_samples):
    score = 3
    if infra[i] == 'piste_separee': score += 1
    if infra[i] == 'rien': score -= 1
    if vitesse[i] == '30': score += 1
    if vitesse[i] == '80': score -= 1
    if pente[i] == 'faible': score -= 1
    if voies[i] == 'multi_voies': score -= 1
    if espace_vert[i] == 'oui': score += 0.5
    note_finale = int(np.clip(np.round(score + np.random.normal(0, 0.5)), 1, 5))
    notes.append(note_finale)

df_photos = pd.read_csv('data\survey\data-photos.csv')
df_base = pd.read_csv('data\survey\data-base.csv')

df_data = pd.merge(df_photos, df_base, on='id_photo')
print(df_data.head())

data = pd.DataFrame({
    'Infrastructure': df_data["type"],
    'Vitesse': df_data["speed"],
    'Pente': df_data["slope"],
    'Nombre_voies':  df_data["nbr_lane"],
    'Espace_vert': df_data["green"],
    'Note': df_data["note"]
})

print("Aperçu des données du sondage :")
print(data.head(), "\n" + "-"*50)

est = HillClimbSearch(scoring_method="k2", max_indegree=4, max_iter=int(1e4))
est.fit(data)

nx.draw_networkx(est.causal_graph_, with_labels=True, node_color='skyblue', node_size=2000, arrowsize=20)
plt.show()

model_MLE = DiscreteBayesianNetwork([
    ('Infrastructure', 'Note'),
    ('Vitesse', 'Note'),
    ('Pente', 'Note'),
    ('Nombre_voies', 'Note'),
    ('Espace_vert', 'Note')
])

model_MLE.fit(data, estimator=DiscreteMLE())
print(model_MLE.check_model())

inference_MLE = VariableElimination(model_MLE)

print("Scénario A : Piste séparée & Vitesse 30 km/h")
resultat_A = inference_MLE.query(variables=['Note'], evidence={'Infrastructure': 3, 'Vitesse': 0, 'Pente':0,'Nombre_voies': 0, 'Espace_vert': 1})
print(resultat_A)

print("\nScénario B : Aucun aménagement & Vitesse 50 km/h")
resultat_B = inference_MLE.query(variables=['Note'], evidence={'Infrastructure': 1, 'Vitesse': 3, 'Pente':0,'Nombre_voies': 3, 'Espace_vert': 0})
print(resultat_B)

model_bayesian = DiscreteBayesianNetwork([
    ('Infrastructure', 'Note'),
    ('Vitesse', 'Note'),
    ('Pente', 'Note'),
    ('Nombre_voies', 'Note'),
    ('Espace_vert', 'Note')
])

model_bayesian.fit(data,DiscreteBayesianEstimator(prior_type="K2"))
print(model_bayesian.check_model())
inference_B = VariableElimination(model_bayesian)

print("Scénario A : Piste séparée & Vitesse 30 km/h")
resultat_A = inference_B.query(variables=['Note'], evidence={'Infrastructure': 3, 'Vitesse': 0, 'Pente':0,'Nombre_voies': 0, 'Espace_vert': 1})
print(resultat_A)

print("\nScénario B : Aucun aménagement & Vitesse 50 km/h")
resultat_B = inference_B.query(variables=['Note'], evidence={'Infrastructure': 1, 'Vitesse': 3, 'Pente':0,'Nombre_voies': 3, 'Espace_vert': 0})
print(resultat_B)
