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
from utils_latent_class import *

# Data preprocessing
base_csv = "data/survey/data-base.csv"
personnes_csv = "data/survey/data-personnes.csv"
photos_csv = "data/survey/data-photos.csv"

df = lc_load_and_prepared_data(base_csv, personnes_csv, photos_csv)
df_personne = lc_transform_df_for_lc(df)

df_personne_enc = lc_encode_df_for_lc(df_personne)

# X y split
y = df_personne_enc[["note_mean"]].copy()
X = df_personne_enc.drop(col_stat + col_note + ['id_personne'], axis=1).copy()

# definition et fit du StepMix model
n_classes = 3

stepmix_model = StepMix(
    n_components=n_classes,
    n_steps=3,
    random_state=RANDOM_STATE,
    measurement="categorical",
    verbose=True
)
stepmix_model.fit(X, y)

# prediction de la class pour chaque réponses
X["class"] = stepmix_model.predict(X)
df_personne["class"] = X["class"]

plot_notes_by_class(df_personne)
plot_notes_by_class(df_personne, 1)
plot_notes_by_class(df_personne, 2)
plot_notes_by_class(df_personne, 6)
# # changement d'encodage pour la regression linéaire
# encoder = OneHotEncoder(dtype="int",sparse_output=False, drop= "first")
# X_to_encode = X_train.loc[:, X_train.columns != 'class']
# array_enc = encoder.fit_transform(X_to_encode)
# encoded_cols = encoder.get_feature_names_out(X_to_encode.columns)
# X_train_enc = pd.DataFrame(array_enc, columns=encoded_cols, index=X_train.index)
# X_train_enc["class"] = X_train["class"]
#
# X_to_encode = X_test.loc[:, X_test.columns != 'class']
# array_enc = encoder.fit_transform(X_to_encode)
# encoded_cols = encoder.get_feature_names_out(X_to_encode.columns)
# X_test_enc = pd.DataFrame(array_enc, columns=encoded_cols, index=X_test.index)
# X_test_enc["class"] = X_test["class"]
#
# # regression pour chaque class
# for nbr_class in range(n_classes):
#     X_train_class = X_train_enc[X_train_enc["class"] == nbr_class].drop(["class"], axis=1)
#     X_test_class = X_test_enc[X_test_enc["class"] == nbr_class].drop(["class"], axis=1)
#     y_train_class = y_train[y_train.index.isin(X_train_class.index)]
#     y_test_class = y_test[y_test.index.isin(X_test_class.index)]
#
#     model = LinearRegression()
#     model.fit(X_train_class, y_train_class)
#
#     coef = model.coef_
#     df_coef = pd.DataFrame({
#         'Features': X_train_class.columns,
#         'coef': coef})
#
#     y_train_pred = model.predict(X_train_class)
#     y_test_pred = model.predict(X_test_class)
#
#     r2_train = r2_score(y_train_class, y_train_pred)
#     r2_test = r2_score(y_test_class, y_test_pred)
#
#     mse_train = mean_squared_error(y_train_class, y_train_pred)
#     mse_test = mean_squared_error(y_test_class, y_test_pred)
#
#     print(f"CLASS {nbr_class}: ")
#
#     print(df_coef.to_string())
#     print(f"R2 train: {r2_train:.4f}, R2 test: {r2_test:.4f}")
#     print(f"MSE train: {mse_train:.4f}, MSE test: {mse_test:.4f}")
#
# #Regression pour toutes les classes (baseline)
#
# X_train_enc.drop(["class"], axis=1, inplace=True)
# X_test_enc.drop(["class"], axis=1, inplace=True)
# model = LinearRegression()
# model.fit(X_train_enc, y_train)
#
# coef = model.coef_
# df_coef = pd.DataFrame({
#         'Features': X_train_enc.columns,
#         'coef': coef})
# y_train_pred = model.predict(X_train_enc)
# y_test_pred = model.predict(X_test_enc)
#
# r2_train = r2_score(y_train, y_train_pred)
# r2_test = r2_score(y_test, y_test_pred)
#
# mse_train = mean_squared_error(y_train, y_train_pred)
# mse_test = mean_squared_error(y_test, y_test_pred)
#
# print(f"ALL CLASSES: ")
#
# print(df_coef.to_string())
# print(f"R2 train: {r2_train:.4f}, R2 test: {r2_test:.4f}")
# print(f"MSE train: {mse_train:.4f}, MSE test: {mse_test:.4f}")