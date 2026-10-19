from utils_benchmark import *
import shap
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

# Global Variables
n_splits = 5
save = True
output_dir = "outputs/model_results/benchmark"
base_csv = "data/survey/data-base.csv"
personnes_csv = "data/survey/data-personnes.csv"
photos_csv = "data/survey/data-photos.csv"


def main():
    X, y, df, dims, coords = load_and_prepare_data(base_csv, personnes_csv, photos_csv)
    models_ML = get_models_ML()
    name = "_student_lyon"
    os.makedirs(output_dir, exist_ok=True)

    # 1. TRAIN / TEST SPLIT
    print("--- Benchmark : Train / Test Split ---")

    print("\n --> ML")
    df_tt, preds_train_tt, y_train_tt, preds_test_tt, y_test_tt = run_train_test_ML(
        models_ML, X, y, random_state=RANDOM_STATE
    )

    print("\n --> MCLR")
    # Retrieve the DataFrame AND prediction dictionaries (modes)
    df_results, test_probabilities, train_probabilities, preds_train_mclr, preds_test_mclr = run_train_test_MCLR(
        df, dims, random_state=RANDOM_STATE
    )


    print(test_probabilities["MCLR Model 1"].head().to_string())
    test_probabilities["MCLR Model 1"].to_csv(os.path.join(output_dir, f"test_probabilities{name}_model1.csv"))
    test_probabilities["MCLR Model 1"].to_csv(os.path.join(output_dir, f"train_probabilities{name}_model1.csv"))
    plot_probabilistic_confusion_matrix(test_probabilities["MCLR Model 1"], name="Test MCLR Model 1")

    print(test_probabilities["MCLR Model 2"].head().to_string())
    test_probabilities["MCLR Model 2"].to_csv(os.path.join(output_dir, f"test_probabilities{name}_model2.csv"))
    test_probabilities["MCLR Model 2"].to_csv(os.path.join(output_dir, f"train_probabilities{name}_model2.csv"))
    plot_probabilistic_confusion_matrix(test_probabilities["MCLR Model 2"], name="Test MCLR Model 2")

    print(train_probabilities["MCLR Model 1"].head().to_string())
    train_probabilities["MCLR Model 1"].to_csv(
        os.path.join(output_dir, f"train_probabilities{name}_model1.csv"))
    train_probabilities["MCLR Model 1"].to_csv(os.path.join(output_dir, f"train_probabilities{name}_model1.csv"))
    plot_probabilistic_confusion_matrix(train_probabilities["MCLR Model 1"], name="Train MCLR Model 1")

    print(train_probabilities["MCLR Model 2"].head().to_string())
    train_probabilities["MCLR Model 2"].to_csv(
        os.path.join(output_dir, f"train_probabilities{name}_model2.csv"))
    train_probabilities["MCLR Model 2"].to_csv(os.path.join(output_dir, f"train_probabilities{name}_model2.csv"))
    plot_probabilistic_confusion_matrix(train_probabilities["MCLR Model 2"], name="Train MCLR Model 2")

    df_results_all = pd.concat([df_tt, df_results])
    print(df_results_all.to_string())
    df_results_all.to_csv(os.path.join(output_dir, f"results_train_test{name}.csv"))

    # --- VISUALIZATIONS ---
    all_preds_train = {**preds_train_tt, **preds_train_mclr}
    all_preds_test = {**preds_test_tt, **preds_test_mclr}

    print("\n --> Generating combined plots (ML + MCLR)...")
    plot_confusion_matrices(y_train_tt, all_preds_train, title_suffix=f"TrainTest_TRAIN{name}",
                            save=save, nrows=2, ncols=4)
    plot_confusion_matrices(y_test_tt, all_preds_test, title_suffix=f"TrainTest_TEST{name}",
                            save=save, nrows=2, ncols=4)

    plot_predictions_distribution(y_train_tt, all_preds_train, title_suffix=f"TrainTest_TRAIN{name}", save=save)
    plot_predictions_distribution(y_test_tt, all_preds_test, title_suffix=f"TrainTest_TEST{name}", save=save)

    # 2. SHUFFLE SPLIT
    print(f"--- Benchmark : ShuffleSplit ({n_splits}-fold) ---")

    print("\n --> ML")
    df_ss_ml, preds_train_ss_ml, y_train_ss_ml, preds_test_ss_ml, y_test_ss_ml = run_shufflesplit_ML(
        models_ML, X, y, n_splits=n_splits, random_state=RANDOM_STATE
    )

    print("\n --> MCLR")
    df_ss_mclr, preds_train_ss_mclr, _, preds_test_ss_mclr, _ = run_shufflesplit_MCLR(
        df, dims, n_splits=n_splits, random_state=RANDOM_STATE
    )

    # --- MERGING METRIC RESULTS ---
    df_ss_all = pd.concat([df_ss_ml, df_ss_mclr], axis=0)
    print(df_ss_all.to_string())
    df_ss_all.to_csv(os.path.join(output_dir, f"results_shufflesplit{name}.csv"))
    all_preds_train_ss = {**preds_train_ss_ml, **preds_train_ss_mclr}
    all_preds_test_ss = {**preds_test_ss_ml, **preds_test_ss_mclr}

    # --- VISUALIZATIONS ---
    print("\n --> Generating combined plots (ShuffleSplit)...")

    plot_confusion_matrices(y_train_ss_ml, all_preds_train_ss,
                            title_suffix=f"ShuffleSplit_TRAIN{name}", save=save, nrows=2, ncols=4)
    plot_confusion_matrices(y_test_ss_ml, all_preds_test_ss,
                            title_suffix=f"ShuffleSplit_TEST{name}", save=save, nrows=2, ncols=4)

    plot_predictions_distribution(y_train_ss_ml, all_preds_train_ss, title_suffix=f"ShuffleSplit_TRAIN{name}",
                                  save=save)
    plot_predictions_distribution(y_test_ss_ml, all_preds_test_ss, title_suffix=f"ShuffleSplit_TEST{name}", save=save)
    print("\n")

    print("\n --> Génération de l'analyse SHAP (HistGradient Boost)...")
    run_shap_analysis(X, y, model_name="HistGradient Boost")

def save_models():
    filepath_save = "data/models/"
    model_ML = get_models_ML()
    X, y, _, _, _ = load_and_prepare_data(base_csv, personnes_csv, photos_csv)
    for model_name, model in model_ML.items():
        train_save_unique_model(model, X, y, model_name, filepath=filepath_save)


def opti_randomized(n_iter = 10):
    X, y, df, dims, coords = load_and_prepare_data(base_csv, personnes_csv, photos_csv)
    models_ML = get_models_ML()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    param_distributions = {
        "Classification Tree": {
            'max_depth': randint(3, 15),
            'min_samples_split': randint(2, 10),
            'min_samples_leaf': randint(1, 10)
        },
        "Gradient Boost": {
            'n_estimators': randint(50, 300),
            'learning_rate': uniform(0.01, 0.2),
            'max_depth': randint(1, 15),
            'subsample': uniform(0.6, 0.4)
        },
        "HistGradient Boost": {
            'learning_rate': uniform(0.01, 0.2),
            'max_iter': randint(50, 300),
            'max_depth': randint(1, 15),
            'l2_regularization': uniform(0.0, 1.0)
        },
        "Random Forest": {
            'n_estimators': randint(50, 300),
            'max_depth': randint(1, 15),
            'min_samples_split': randint(2, 10),
            'min_samples_leaf': randint(1, 10)
        },
        "Ada Boost": {
            'n_estimators': randint(50, 300),
            'learning_rate': uniform(0.01, 1.0)
        },
        "ExtraTrees": {
            'n_estimators': randint(50, 300),
            'max_depth': randint(1, 15),
            'min_samples_split': randint(2, 10),
            'min_samples_leaf': randint(1, 10)
        }
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    results_summary = {}

    for name, model in models_ML.items():
        if name not in param_distributions:
            continue
        print(f"Randomized Search for model : {name}")

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions[name],
            n_iter=n_iter,
            scoring='balanced_accuracy',
            cv=cv,
            n_jobs=-1,
            random_state=RANDOM_STATE
        )

        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        safe_name = name.lower().replace(" ", "_")
        joblib.dump(best_model, f"outputs/model_results/models/{safe_name}_optimized.joblib")

        train_preds = best_model.predict(X_train)
        test_preds = best_model.predict(X_test)

        train_metrics = get_metrics_ML(y_train, train_preds)
        test_metrics = get_metrics_ML(y_test, test_preds)

        results_summary[name] = {
            'best_params': search.best_params_,
            'best_score_cv': search.best_score_,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'model': best_model
        }

    return results_summary


def run_shap_on_loaded_models():
    # 1. Charger les données en utilisant vos utilitaires existants[cite: 1, 2]
    X, y, df, dims, coords = load_and_prepare_data(base_csv, personnes_csv, photos_csv)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    filepath_load = "outputs/model_results/models/"
    output_plot_dir = "outputs/model_results/benchmark/plots/"
    os.makedirs(output_plot_dir, exist_ok=True)

    if not os.path.exists(filepath_load):
        print(f"Le dossier {filepath_load} est introuvable.")
        return

    # 2. Parcourir tous les fichiers du dossier de modèles
    for filename in os.listdir(filepath_load):
        if filename.endswith(".joblib") and "optimized" in filename:
            model_path = os.path.join(filepath_load, filename)
            model_name = filename.replace("_optimized.joblib", "").replace("_", " ").title()

            print(f"\n-> Chargement du modèle : {model_name}")
            model = joblib.load(model_path)

            try:
                print(f"-> Calcul des valeurs SHAP pour {model_name}...")
                masker = shap.maskers.Independent(X_train, max_samples=X_train.shape[0])
                explainer = shap.Explainer(model.predict_proba, masker)
                shap_values = explainer(X_test)

                # 3. Génération et sauvegarde du graphique global (Summary Plot)
                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_values, X_test, show=False)
                plt.title(f"Impact global SHAP - {model_name}", fontsize=14, fontweight='bold', pad=15)
                plt.tight_layout()

                save_path = os.path.join(output_plot_dir, f"shap_summary_{filename.replace('.joblib', '')}.png")
                plt.savefig(save_path, bbox_inches='tight')
                plt.close()
                print(f"-> Graphique sauvegardé dans : {save_path}")

            except Exception as e:
                print(f"Erreur lors de l'analyse SHAP pour {filename} : {e}")

if __name__ == "__main__":
    # main()
    # results_summary = opti_randomized(n_iter=2000)
    # results_summary = pd.DataFrame(results_summary)
    # results_summary.to_csv(os.path.join(output_dir, "results_summary.csv"))

    run_shap_on_loaded_models()
    # save_models()