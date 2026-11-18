from stepmix.stepmix import StepMix
from utils_benchmark import *


base_csv = "data/survey/data-base.csv"
personnes_csv = "data/survey/data-personnes.csv"
photos_csv = "data/survey/data-photos.csv"
X, y = load_and_prepare_data(base_csv, personnes_csv, photos_csv, people=True)

col_covariate = ['age_1','age_2','age_3','gender_0','gender_1','gender_2','electric_bike_True', 'electric_bike_False',
                  'bike_use_frequency_0','bike_use_frequency_1','bike_use_frequency_2','bike_use_frequency_3',
                  'bike_use_frequency_4','bike_ownership_0','bike_ownership_1']

Y = y.values.reshape(-1, 1)
Z_cov = X[col_covariate].values

model = StepMix(
    n_components=4,
    n_steps=1,
    measurement="continuous",
    structural="covariate",
    verbose=1,
    random_state=42
)

model.fit(Y, Z_cov)

class_shares = model.weights_
print("Parts estimées des classes :", class_shares)
X['predicted_class'] = model.predict(Y, Z_cov)

report_means = X.groupby('predicted_class')[col_covariate].mean()

print("--- Rapport des moyennes des covariables par classe ---")
print(report_means.T)