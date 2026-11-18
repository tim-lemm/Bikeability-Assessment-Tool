from stepmix.stepmix import StepMix
from utils_benchmark import load_and_prepare_data
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd


base_csv = "data/survey/data-base.csv"
personnes_csv = "data/survey/data-personnes.csv"
photos_csv = "data/survey/data-photos.csv"
X, y = load_and_prepare_data(base_csv, personnes_csv, photos_csv, people=True, drop="first")

col_covariate = ['gender_0','gender_1','gender_2','electric_bike_True',
                  'bike_use_frequency_0','bike_use_frequency_1','bike_use_frequency_2','bike_use_frequency_3',
                  'bike_use_frequency_4','bike_ownership_1']
col_predictors = [ 'type_1', 'type_2', 'type_3', 'slope_1','slope_2',"green_1","green_2",
                  'nbr_lane_1','nbr_lane_2', 'speed_1', 'speed_2', 'speed_3']

Y = y.values.reshape(-1, 1)
Z_cov = X[col_covariate]

# model = StepMix(
#     n_components=4,
#     n_steps=3,
#     measurement="continuous",
#     structural="covariate",
#     # verbose=1,
#     random_state=42
# )
#
#
# grid = {
#     'n_components': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
# }
# gs = GridSearchCV(estimator=model, cv=3, param_grid=grid)
# gs.fit(Y, Z_cov)
#
# results = pd.DataFrame(gs.cv_results_)
# results["Val. Log Likelihood"] = results['mean_test_score']
#
# print(results.to_string())
#
# sns.set_style("darkgrid")
# sns.lineplot(data=results, x='param_n_components', y='Val. Log Likelihood',
#              hue='param_n_steps', palette='Dark2')
# plt.show()

model = StepMix(
    n_components=4,
    n_steps=3,
    measurement="binary",
    structural="covariate",
    verbose=1,
    random_state=42
)

print(f"\n === CLASSES ===")

model.fit(Y, Z_cov)

X['predicted_class'] = model.predict(Y, Z_cov)
report_means = X.groupby('predicted_class')[col_covariate].mean()

print("--- Rapport des moyennes des covariables par classe ---")
print(report_means.T)

# print(f"\n === REGRESSION ===")
#
# for class_id in sorted(X['predicted_class'].unique()):
#     print("="*80)
#     print(f"--- Classe {class_id} ---")
#     print("=" * 80)
#     df_class = X[X['predicted_class'] == class_id].copy()
#
#     Z_pred = df_class[col_predictors].copy()
#     Z_pred = Z_pred.loc[:, Z_pred.nunique() > 1]
#     Z_pred = sm.add_constant(Z_pred)
#     Y_class = y.loc[Z_pred.index]
#
#     regression_model = sm.OLS(Y_class, Z_pred).fit()
#
#     print(regression_model.summary())