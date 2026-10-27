from utils_benchmark import *
from sklearn.model_selection import RandomizedSearchCV, StratifiedShuffleSplit
import joblib

# Global Variables
n_splits = 5
save = True
output_dir = "outputs/photo_people/model_results/"
base_csv = "data/survey/data-base.csv"
personnes_csv = "data/survey/data-personnes.csv"
photos_csv = "data/survey/data-photos.csv"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir + "/models/", exist_ok=True)


X, y = load_and_prepare_data(base_csv, personnes_csv, photos_csv, people=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=RANDOM_STATE)

models_config = get_models_ML()

all_results = []
trained_models = {}

print("BENCHMARK - START\n")

for name, config in models_config.items():
  print(f"--- Optimisation et évaluation pour : {name} ---")

  search = RandomizedSearchCV(
      estimator=config["model"],
      param_distributions=config["params"],
      n_iter=1000,
      cv=cv,
      scoring="balanced_accuracy",
      random_state=RANDOM_STATE,
      n_jobs=-1,
  )
  search.fit(X_train, y_train)

  best_model = search.best_estimator_
  trained_models[name] = best_model
  if save:
      joblib.dump(best_model, f'{output_dir}/models/{name.lower().replace(" ", "_")}.joblib')
  y_pred = best_model.predict(X_test)

  acc = accuracy_score(y_test, y_pred)
  bal_acc = balanced_accuracy_score(y_test, y_pred)
  mae = mean_absolute_error(y_test, y_pred)
  prec_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
  qwk = cohen_kappa_score(y_test, y_pred, weights="quadratic")

  result_entry = {
      "Model": name,
      "Best CV Balanced Acc": search.best_score_,
      "Test Accuracy": acc,
      "Test Balanced Accuracy": bal_acc,
      "Test MAE": mae,
      "Test QWK": qwk,
  }

  for idx, prec in enumerate(prec_per_class):
    result_entry[f"Precision_Class_{idx}"] = prec

  all_results.append(result_entry)
  print(f" Fait. Best CV Score : {search.best_score_:.4f}\n")

print("BENCHMARK - END\n")

results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values(
    by="Test Balanced Accuracy", ascending=False
).reset_index(drop=True)

print("=" * 300)
print(" TABLEAU COMPARATIF GLOBAL DES MODÈLES")
print("=" * 300)
print(results_df.to_string(index=False))

if save:
  csv_path = os.path.join(output_dir, "benchmark_metrics_summary.csv")
  results_df.to_csv(csv_path, index=False)
  print(f"\n Résultats sauvegardés dans : {csv_path}")

plot_all_confusion_matrices(trained_models, X_test, y_test, output_dir, save)