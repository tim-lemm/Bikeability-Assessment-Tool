from stepmix.stepmix import StepMix
from utils_benchmark import load_and_prepare_data, import_and_merge_data_base
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import pandas as pd
from config import *

# Data preprocessing
base_csv = "data/survey/data-base.csv"
personnes_csv = "data/survey/data-personnes.csv"
photos_csv = "data/survey/data-photos.csv"

categorical_cols = ["nbr_lane", "speed", "slope", "type", "green", "age", "gender", "job", "bike_use_frequency"]
covariate_cols = ["gender", "job", "age", "bike_use_frequency", "electric_bike", "bike_ownership", "type_bike"]

df = import_and_merge_data_base(base_csv, personnes_csv, photos_csv)
df.drop(["postal_code","type_bike_fold","type_bike_cargo", "type_bike_recumbent", "traffic"], axis=1, inplace=True)
df["electric_bike"] = df["electric_bike"].astype(int)

df["type_bike"] = "city"
df.loc[df["type_bike_allroad"] == 1, "type_bike"] = "allroad"
df.loc[df["type_bike_road"] == 1, "type_bike"] = "road"

df.drop(["type_bike_allroad", "type_bike_city", "type_bike_road"], axis=1, inplace=True)

#transform df from long to large df_personne

df_personne = df.copy()
df_personne.drop(["nbr_lane", "speed", "slope", "green", "type"], axis=1, inplace=True)
cols_personne = [c for c in df_personne.columns if c not in ["id_photo", "id_personne", "note"]]
df_notes = df_personne.pivot(index="id_personne", columns="id_photo", values="note")
df_notes.columns = [f"note_{col}" for col in df_notes.columns]
df_notes.reset_index(inplace=True)

df_carac = df_personne[["id_personne"] + cols_personne].drop_duplicates(subset="id_personne")

df_personne = df_carac.merge(df_notes, on = "id_personne", how = "left")

col_note = [f"note_{i}" for i in range(1,18)]

df_personne["note_mean"] = df_personne[col_note].mean(axis = 1)
df_personne["note_std"] = df_personne[col_note].std(axis = 1)
df_personne["note_min"] = df_personne[col_note].min(axis = 1)
df_personne["note_max"] = df_personne[col_note].max(axis = 1)
df_personne["note_med"] = df_personne[col_note].median(axis = 1)
quantiles = df_personne[col_note].quantile([0.25,0.75], axis = 1)
df_personne["note_q25"] = quantiles.loc[0.25]
df_personne["note_q75"] = quantiles.loc[0.75]

# encodage pour le StepMix model
original_mappings = {}
label_encoder = LabelEncoder()
for col in df.columns:
    df[col] = label_encoder.fit_transform(df[col])
    original_mappings[col] = {i: label for i, label in enumerate(label_encoder.classes_)}

# décommenter pour le dictionaire de l'encodage
# for col in covariate_cols:
#     print(f"\n{col}:")
#     for encoded_val, original_val in original_mappings[col].items():
#         print(f"  Code {encoded_val} -> {original_val}")

# Train Test split
y = df["note"].copy()
X = df.drop(["note"], axis=1).copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE,
                                                            stratify=y)
X_cov_train = X_train[covariate_cols].copy()
X_cov_test = X_test[covariate_cols].copy()

# definition et fit du StepMix model
n_classes = 3

stepmix_model = StepMix(
    n_components=n_classes,
    n_steps=3,
    random_state=RANDOM_STATE,
    measurement="categorical",
    verbose=True
)
stepmix_model.fit(X_cov_train, y_train)

# prediction de la class pour chaque réponses
X_train["class"] = stepmix_model.predict(X_cov_train)
X_test["class"] = stepmix_model.predict(X_cov_test)

print(pd.crosstab(X_train["bike_use_frequency"], X_train["class"]))

# changement d'encodage pour la regression linéaire
encoder = OneHotEncoder(dtype="int",sparse_output=False, drop= "first")
X_to_encode = X_train.loc[:, X_train.columns != 'class']
array_enc = encoder.fit_transform(X_to_encode)
encoded_cols = encoder.get_feature_names_out(X_to_encode.columns)
X_train_enc = pd.DataFrame(array_enc, columns=encoded_cols, index=X_train.index)
X_train_enc["class"] = X_train["class"]

X_to_encode = X_test.loc[:, X_test.columns != 'class']
array_enc = encoder.fit_transform(X_to_encode)
encoded_cols = encoder.get_feature_names_out(X_to_encode.columns)
X_test_enc = pd.DataFrame(array_enc, columns=encoded_cols, index=X_test.index)
X_test_enc["class"] = X_test["class"]

# regression pour chaque class
for nbr_class in range(n_classes):
    X_train_class = X_train_enc[X_train_enc["class"] == nbr_class].drop(["class"], axis=1)
    X_test_class = X_test_enc[X_test_enc["class"] == nbr_class].drop(["class"], axis=1)
    y_train_class = y_train[y_train.index.isin(X_train_class.index)]
    y_test_class = y_test[y_test.index.isin(X_test_class.index)]

    model = LinearRegression()
    model.fit(X_train_class, y_train_class)

    coef = model.coef_
    df_coef = pd.DataFrame({
        'Features': X_train_class.columns,
        'coef': coef})

    y_train_pred = model.predict(X_train_class)
    y_test_pred = model.predict(X_test_class)

    r2_train = r2_score(y_train_class, y_train_pred)
    r2_test = r2_score(y_test_class, y_test_pred)

    mse_train = mean_squared_error(y_train_class, y_train_pred)
    mse_test = mean_squared_error(y_test_class, y_test_pred)

    print(f"CLASS {nbr_class}: ")

    print(df_coef.to_string())
    print(f"R2 train: {r2_train:.4f}, R2 test: {r2_test:.4f}")
    print(f"MSE train: {mse_train:.4f}, MSE test: {mse_test:.4f}")

#Regression pour toutes les classes (baseline)

X_train_enc.drop(["class"], axis=1, inplace=True)
X_test_enc.drop(["class"], axis=1, inplace=True)
model = LinearRegression()
model.fit(X_train_enc, y_train)

coef = model.coef_
df_coef = pd.DataFrame({
        'Features': X_train_enc.columns,
        'coef': coef})
y_train_pred = model.predict(X_train_enc)
y_test_pred = model.predict(X_test_enc)

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

print(f"ALL CLASSES: ")

print(df_coef.to_string())
print(f"R2 train: {r2_train:.4f}, R2 test: {r2_test:.4f}")
print(f"MSE train: {mse_train:.4f}, MSE test: {mse_test:.4f}")