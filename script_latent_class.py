from stepmix.stepmix import StepMix
from config import *
from utils_latent_class import *
from sklearn.linear_model import LinearRegression, RidgeCV, LogisticRegression, Perceptron, SGDClassifier, Lasso
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, precision_score, accuracy_score
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeClassifier

# Data preprocessing
base_csv = "data/survey/data-base.csv"
personnes_csv = "data/survey/data-personnes.csv"
photos_csv = "data/survey/data-photos.csv"

df = lc_load_and_prepared_data(base_csv, personnes_csv, photos_csv)
df_personne = lc_transform_df_for_lc(df)

df_personne_enc = lc_encode_df_for_lc(df_personne)
#TODO scale first the features

# X y split
y = df_personne_enc[["note_mean"]].copy()
X = df_personne_enc.drop(col_stat + col_note + ['id_personne'], axis=1).copy()

# definition et fit du StepMix model
n_classes = 4

stepmix_model = StepMix(
    n_components=n_classes,
    n_steps=3,
    random_state=RANDOM_STATE,
    measurement="continuous",
    verbose=True
)
stepmix_model.fit(X, y)

kmeans_model = KMeans(
    n_clusters=n_classes,
    random_state=RANDOM_STATE,
    n_init="auto",
)
kmeans_model.fit(X)

#TODO finish kmeans implementation

# prediction de la class pour chaque réponses
X["class"] = stepmix_model.predict(X)
df_personne["class"] = X["class"]

other_cols = [c for c in df_personne.columns if c not in col_note and c not in col_stat]
long_df = df_personne.melt(
    id_vars=other_cols,
    value_vars=col_note,
    var_name="col_note",
    value_name="note",
)
long_df["id_photo"] = long_df["col_note"].str.replace("note_", "", regex=False).astype(int)
long_df.drop(["col_note"], axis=1, inplace=True)
# 2. DataFrame des statistiques par (class, note_id)
stats_df = (
    long_df.groupby(["class", "id_photo"], as_index=False)["note"]
    .agg(
        count="count",
        mean="mean",
        std="std",
        min="min",
        q25=lambda s: np.percentile(s, 25),
        median="median",
        q75=lambda s: np.percentile(s, 75),
        max="max",
    )
    .sort_values(["class", "id_photo"])
    .reset_index(drop=True)
)

print(stats_df.to_string())

df = df.merge(df_personne[["class", "id_personne"]], on = "id_personne", how = "left")
X = df.drop(["note", "id_personne", 'id_photo'], axis=1).copy()
y = df["note"]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

# changement d'encodage pour la regression linéaire
encoder = OneHotEncoder(dtype="int",sparse_output=False, drop= "first")
X_to_encode = X_train.loc[:, X_train.columns != 'class']


X_train_enc_arr = encoder.fit_transform(X_train.drop(columns=["class"]))
encoded_cols = encoder.get_feature_names_out(X_to_encode.columns)
X_train_enc = pd.DataFrame(X_train_enc_arr, columns=encoded_cols, index=X_train.index)
X_train_enc["class"] = X_train["class"]

X_test_enc_arr = encoder.transform(X_test.drop(columns=["class"]))
X_test_enc = pd.DataFrame(X_test_enc_arr, columns=encoded_cols, index=X_test.index)
X_test_enc["class"] = X_test["class"]


max_depth = 3
# regression pour chaque class
for nbr_class in range(n_classes):
    X_train_class = X_train_enc[X_train_enc["class"] == nbr_class].drop(["class"], axis=1)
    X_test_class = X_test_enc[X_test_enc["class"] == nbr_class].drop(["class"], axis=1)
    y_train_class = y_train.loc[X_train_class.index]
    y_test_class = y_test.loc[X_test_class.index]

    model = LinearRegression()
    model.fit(X_train_class, y_train_class)

    # coef = model.coef_
    # df_coef = pd.DataFrame({
    #     'Features': X_train_class.columns,
    #     'coef': coef})

    y_train_pred = model.predict(X_train_class)
    y_test_pred = model.predict(X_test_class)

    r2_train = r2_score(y_train_class, y_train_pred)
    r2_test = r2_score(y_test_class, y_test_pred)

    acc_train = accuracy_score(y_train_class, np.rint(y_train_pred))
    acc_test = accuracy_score(y_test_class, np.rint(y_test_pred))

    mse_train = mean_squared_error(y_train_class, y_train_pred)
    mse_test = mean_squared_error(y_test_class, y_test_pred)

    print(f"CLASS {nbr_class}: ")

    # print(df_coef.to_string())
    print(f"R2 train: {r2_train:.4f}, R2 test: {r2_test:.4f}")
    print(f"Accuracy score train: {acc_train:.4f}, Accuracy test: {acc_test:.4f}")
    print(f"MSE train: {mse_train:.4f}, MSE test: {mse_test:.4f}")

#Regression pour toutes les classes (baseline)

X_train_enc.drop(["class"], axis=1, inplace=True)
X_test_enc.drop(["class"], axis=1, inplace=True)
model = LinearRegression()
model.fit(X_train_enc, y_train)

# coef = model.coef_
# df_coef = pd.DataFrame({
#         'Features': X_train_enc.columns,
#         'coef': coef})
y_train_pred = model.predict(X_train_enc)
y_test_pred = model.predict(X_test_enc)

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

acc_train = accuracy_score(y_train, np.rint(y_train_pred))
acc_test = accuracy_score(y_test, np.rint(y_test_pred))

mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

print(f"ALL CLASSES: ")

# print(df_coef.to_string())
print(f"R2 train: {r2_train:.4f}, R2 test: {r2_test:.4f}")
print(f"Accuracy score train: {acc_train:.4f}, accuracy test: {acc_test:.4f}")
print(f"MSE train: {mse_train:.4f}, MSE test: {mse_test:.4f}")