from stepmix.stepmix import StepMix
from utils_benchmark import load_and_prepare_data, import_and_merge_data_base
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
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

df = import_and_merge_data_base(base_csv, personnes_csv, photos_csv)
df.drop(["postal_code","type_bike_fold","type_bike_cargo", "type_bike_recumbent", "id_photo", "id_personne", "traffic"], axis=1, inplace=True)
df["electric_bike"] = df["electric_bike"].astype(int)


y = df["note"].copy()
X = df.drop(["note"], axis=1).copy()




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(drop="first", dtype="int", sparse_output=False), categorical_cols),
],
    remainder="passthrough"
    )

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression()),
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE : {mse:.4f}")
print(f" R² : {r2:.4f}")

encoded_cat_cols = (
    model.named_steps['preprocessor']
    .named_transformers_['cat']
    .get_feature_names_out(categorical_cols)
.tolist()
)
binary_cols = [col for col in X.columns if col not in categorical_cols]
all_feature_names = encoded_cat_cols + binary_cols

coefficients = pd.DataFrame({
    'Variable': all_feature_names,
    'Coefficient': model.named_steps['regressor'].coef_
})
print("\n Model coefficients :")
print(coefficients.to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Actual vs. Predicted (Parity Plot)
axes[0].scatter(y_test, y_pred, alpha=0.5, color='royalblue')
axes[0].set_title("Actual vs. Predicted Values")
axes[0].set_xlabel("Actual Note")
axes[0].set_ylabel("Predicted Note")

# Plot 2: Residual Plot
residuals = y_test - y_pred
axes[1].scatter(y_pred, residuals, alpha=0.5, color='darkorange')
axes[1].axhline(0,linestyle='--', lw=2)
axes[1].set_title("Residual Plot")
axes[1].set_xlabel("Predicted Note")
axes[1].set_ylabel("Residuals (Actual - Predicted)")

plt.tight_layout()
plt.show()