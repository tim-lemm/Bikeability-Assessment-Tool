from utils_benchmark import run_train_test, run_shufflesplit, run_loo, load_and_prepare_data, get_models, get_models_gb, plot_confusion_matrices
import pandas as pd
import os
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

random_state=42
n_splits=5

def main():
    base_csv = "data/survey/data-base.csv"
    personnes_csv = "data/survey/data-personnes.csv"
    photos_csv = "data/survey/data-photos.csv"

    X, y = load_and_prepare_data(base_csv, personnes_csv, photos_csv)
    models = get_models_gb()
    name = "_gb"
    output_dir = "outputs/model_results/benchmark"
    os.makedirs(output_dir, exist_ok=True)

    # 1. TRAIN / TEST SPLIT
    print("--- Benchmark : Train / Test Split ---")
    df_tt, preds_train_tt, y_train_tt, preds_test_tt, y_test_tt = run_train_test(models, X, y, random_state=random_state)
    print(df_tt.to_string())
    df_tt.to_csv(os.path.join(output_dir, f"results_train_test{name}.csv"))

    plot_confusion_matrices(y_train_tt, preds_train_tt, title_suffix=f"TrainTest_TRAIN{name}", save=True)
    plot_confusion_matrices(y_test_tt, preds_test_tt, title_suffix=f"TrainTest_TEST{name}", save=True)
    print("\n")

    # 2. SHUFFLE SPLIT
    print(f"--- Benchmark : ShuffleSplit ({n_splits}-fold) ---")
    df_ss, preds_train_ss, y_train_ss, preds_test_ss, y_test_ss = run_shufflesplit(models, X, y, n_splits=n_splits, random_state=random_state)
    print(df_ss.to_string())
    df_ss.to_csv(os.path.join(output_dir, f"results_shufflesplit{name}.csv"))

    plot_confusion_matrices(y_train_ss, preds_train_ss, title_suffix=f"ShuffleSplit_TRAIN{name}", save=True)
    plot_confusion_matrices(y_test_ss, preds_test_ss, title_suffix=f"ShuffleSplit_TEST{name}", save=True)
    print("\n")

    # 3. LEAVEONE OUT (LOO)
    # print("--- Benchmark : Leave-One-Out (LOO) ---")
    # df_loo, preds_train_loo, y_train_loo, preds_test_loo, y_loo = run_loo(models, X, y)
    # print(df_loo.to_string())
    # df_loo.to_csv(os.path.join(output_dir, f"results_loo{name}.csv"))
    #
    # plot_confusion_matrices(y_train_loo, preds_train_loo, title_suffix=f"LOO_TRAIN{name}", save=True)
    # plot_confusion_matrices(y_loo, preds_test_loo, title_suffix=f"LOO_TEST{name}", save=True)


def optimize_gradient_boosting(X, y):
    # 1. Définition du modèle de base
    gb = GradientBoostingClassifier(random_state=42)
    cv = 5
    # 2. Définition de la grille d'hyperparamètres à tester
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.005, 0.01, 0.05, 0.1, 0.2, 0.3],
        'max_depth': [1, 2, 3, 4, 5],
        'min_samples_split': [2, 5, 10, 15],
        'subsample': [0.5, 0.8, 1.0],
        'min_samples_leaf':[1,2,3]
    }

    # 3. Choix du score à optimiser
    scoring_metric = 'accuracy'

    print("Lancement de la recherche des hyperparamètres optimaux (GridSearchCV)...")

    # 4. Configuration du GridSearch (ici en 5-fold cross-validation)
    grid_search = GridSearchCV(
        estimator=gb,
        param_grid=param_grid,
        scoring=scoring_metric,
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    # 5. Exécution de la recherche
    grid_search.fit(X, y)

    # 6. Affichage des résultats
    print("\n--- Résultats de l'optimisation ---")
    print(f"Meilleur score ({scoring_metric}) : {grid_search.best_score_:.4f}")
    print("Meilleurs hyperparamètres trouvés :")
    for param, value in grid_search.best_params_.items():
        print(f"  -> {param}: {value}")

    return grid_search.best_estimator_


def main_2():
    base_csv = "data/survey/data-base.csv"
    personnes_csv = "data/survey/data-personnes.csv"
    photos_csv = "data/survey/data-photos.csv"

    X, y = load_and_prepare_data(base_csv, personnes_csv, photos_csv)

    # Lance l'optimisation
    best_model = optimize_gradient_boosting(X, y)


if __name__ == "__main__":
    main()