from sklearn.metrics import accuracy_score, balanced_accuracy_score
from utils_benchmark import *
import pandas as pd
import os
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold

random_state=74
n_splits=5
save = False
output_dir = "outputs/model_results/benchmark"
base_csv = "data/survey/data-base.csv"
personnes_csv = "data/survey/data-personnes.csv"
photos_csv = "data/survey/data-photos.csv"

def main():

    X, y, df, dims, coords = load_and_prepare_data(base_csv, personnes_csv, photos_csv)
    models_ML = get_models_ML()

    name = ""
    os.makedirs(output_dir, exist_ok=True)

    # 1. TRAIN / TEST SPLIT
    print("--- Benchmark : Train / Test Split ---")

    print("\n --> ML")
    df_tt, preds_train_tt, y_train_tt, preds_test_tt, y_test_tt = run_train_test_ML(models_ML, X, y,
                                                                                    random_state=random_state)

    print("\n --> MCLR")
    # On récupère le DataFrame ET les dictionnaires de prédictions (modes)
    df_results, preds_train_mclr, preds_test_mclr = run_train_test_MCLR(df, dims, random_state=random_state)

    # --- FUSION DES RÉSULTATS DE MÉTRIQUES ---
    df_results_all = pd.concat([df_tt, df_results])
    print(df_results_all.to_string())
    df_results_all.to_csv(os.path.join(output_dir, f"results_train_test{name}.csv"))

    # --- VISUALISATIONS ---
    print("Clés présentes dans le ML :", list(preds_test_tt.keys()))
    print("Clés présentes dans le MCLR :", list(preds_test_mclr.keys()))
    all_preds_train = {**preds_train_tt, **preds_train_mclr}
    all_preds_test = {**preds_test_tt, **preds_test_mclr}

    print("\n --> Génération des graphiques combinés (ML + MCLR)...")
    plot_confusion_matrices(y_train_tt, all_preds_train, title_suffix=f"TrainTest_TRAIN{name}", save=save, nrows=2,
                            ncols=4)
    plot_confusion_matrices(y_test_tt, all_preds_test, title_suffix=f"TrainTest_TEST{name}", save=save, nrows=2,
                            ncols=4)

    plot_predictions_distribution(y_train_tt, all_preds_train, title_suffix=f"TrainTest_TRAIN{name}", save=save)
    plot_predictions_distribution(y_test_tt, all_preds_test, title_suffix=f"TrainTest_TEST{name}", save=save)



    # 2. SHUFFLE SPLIT
    print(f"--- Benchmark : ShuffleSplit ({n_splits}-fold) ---")

    print("\n --> ML")
    df_ss_ml, preds_train_ss_ml, y_train_ss_ml, preds_test_ss_ml, y_test_ss_ml = run_shufflesplit_ML(
        models_ML, X, y, n_splits=n_splits, random_state=random_state
    )

    print("\n --> MCLR")
    df_ss_mclr, preds_train_ss_mclr, _, preds_test_ss_mclr, _ = run_shufflesplit_MCLR(
        df, dims, n_splits=n_splits, random_state=random_state
    )

    # --- FUSION DES RÉSULTATS DE MÉTRIQUES ---
    df_ss_all = pd.concat([df_ss_ml, df_ss_mclr], axis=0)
    print(df_ss_all.to_string())
    df_ss_all.to_csv(os.path.join(output_dir, f"results_shufflesplit{name}.csv"))
    all_preds_train_ss = {**preds_train_ss_ml, **preds_train_ss_mclr}
    all_preds_test_ss = {**preds_test_ss_ml, **preds_test_ss_mclr}

    # --- VISUALISATIONS ---
    print("\n --> Génération des graphiques combinés (ShuffleSplit)...")

    plot_confusion_matrices(
        y_train_ss_ml, all_preds_train_ss, title_suffix=f"ShuffleSplit_TRAIN{name}", save=save, nrows=2, ncols=4
    )
    plot_confusion_matrices(
        y_test_ss_ml, all_preds_test_ss, title_suffix=f"ShuffleSplit_TEST{name}", save=save, nrows=2, ncols=4
    )

    plot_predictions_distribution(
        y_train_ss_ml, all_preds_train_ss, title_suffix=f"ShuffleSplit_TRAIN{name}", save=save
    )
    plot_predictions_distribution(
        y_test_ss_ml, all_preds_test_ss, title_suffix=f"ShuffleSplit_TEST{name}", save=save
    )
    print("\n")


def opti():
    base_csv = "data/survey/data-base.csv"
    personnes_csv = "data/survey/data-personnes.csv"
    photos_csv = "data/survey/data-photos.csv"

    X, y = load_and_prepare_data(base_csv, personnes_csv, photos_csv)

    # param_grid = {
    #     'learning_rate': [0.01, 0.05, 0.06, 0.07, 0.1],
    #     'max_depth': [2, 3, 4],
    #     'min_samples_leaf': [7, 10, 20, 50],
    #     'n_estimators': [100, 200, 300]
    # }
    # param_grid = {
    #     'learning_rate': [0.01, 0.05, 0.1, 0.2],
    #     'n_estimators': [100, 200, 300, 500],
    #     'max_depth': [2, 3, 4, 5],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 5, 10, 20],
    #     'subsample': [0.8, 0.9, 1.0]
    # }
    # param_grid = {
    #     'learning_rate': [0.1, 0.2, 0.3],
    #     'max_iter':[20,50,100,200,300],
    #     'max_leaf_nodes':[None, 30, 40],
    #     'max_depth':[None, 5, 10],
    #     'min_samples_leaf':[1, 2, 3, 4, 20],
    #     'max_features':[1,0.99,0.98,0.95]
    # }
    param_grid = {
        'criterion' : ['gini', 'entropy', 'log_loss'],
        'n_estimators' : [50, 100, 150, 200],
        'max_depth':[None, 1,3,5],
        'max_features':['sqrt', 'log2', None],
        'min_samples_leaf':[1, 2, 3, 10, 25],
        'min_samples_split':[2,3,4],
        'bootstrap':[True],
        'oob_score':[accuracy_score, balanced_accuracy_score],
                  }

    model = RandomForestClassifier(random_state=42)
    model_name = "RandomForestClassifier"
    # Lance l'optimisation
    best_model, best_parameter = optimize_model_hp(X, y, model, param_grid, scoring_metric="precision_weighted")
    df_best_parameters = pd.DataFrame([best_parameter])
    print(df_best_parameters.to_string())
    df_best_parameters.to_csv(os.path.join(output_dir, f"results_GS_best_parameter_{model_name}.csv"))
    # Entraine le meilleur modéle
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    best_model.fit(X_train, y_train)
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    # Affichage résultats meilleur modéle
    train_metrics = get_metrics_ML(y_train, y_train_pred)
    test_metrics = get_metrics_ML(y_test, y_test_pred)

    results = {}
    results["best_model"] = {}
    for k, v in train_metrics.items():
        results["best_model"][f'Train {k}'] = v
    for k, v in test_metrics.items():
        results["best_model"][f'Test {k}'] = v

    df_results = pd.DataFrame(results).T
    df_results.to_csv(os.path.join(output_dir, f"results_GS_{model_name}.csv"))
    print(df_results.to_string())

if __name__ == "__main__":
    main()