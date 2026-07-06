from sklearn.metrics import accuracy_score, balanced_accuracy_score
from utils_benchmark import run_train_test, run_shufflesplit, run_loo, load_and_prepare_data, get_models, get_models_gb, plot_confusion_matrices, get_metrics, optimize_model_hp, plot_predictions_distribution
import pandas as pd
import os
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold

random_state=74
n_splits=100
output_dir = "outputs/model_results/benchmark"
base_csv = "data/survey/data-base.csv"
personnes_csv = "data/survey/data-personnes.csv"
photos_csv = "data/survey/data-photos.csv"

def main():

    X, y = load_and_prepare_data(base_csv, personnes_csv, photos_csv)
    models = (get_models())
    name = ""
    os.makedirs(output_dir, exist_ok=True)

    # 1. TRAIN / TEST SPLIT
    print("--- Benchmark : Train / Test Split ---")
    df_tt, preds_train_tt, y_train_tt, preds_test_tt, y_test_tt = run_train_test(models, X, y, random_state=random_state)
    print(df_tt.to_string())
    df_tt.to_csv(os.path.join(output_dir, f"results_train_test{name}.csv"))

    plot_confusion_matrices(y_train_tt, preds_train_tt, title_suffix=f"TrainTest_TRAIN{name}", save=False)
    plot_confusion_matrices(y_test_tt, preds_test_tt, title_suffix=f"TrainTest_TEST{name}", save=False)
    plot_predictions_distribution(y_train_tt, preds_train_tt, title_suffix=f"TrainTest_TRAIN{name}", save=False)
    plot_predictions_distribution(y_test_tt, preds_test_tt, title_suffix=f"TrainTest_TEST{name}", save=False)
    print("\n")

    # # 2. SHUFFLE SPLIT
    # print(f"--- Benchmark : ShuffleSplit ({n_splits}-fold) ---")
    # df_ss, preds_train_ss, y_train_ss, preds_test_ss, y_test_ss = run_shufflesplit(models, X, y, n_splits=n_splits, random_state=random_state)
    # print(df_ss.to_string())
    # df_ss.to_csv(os.path.join(output_dir, f"results_shufflesplit{name}.csv"))
    #
    # plot_confusion_matrices(y_train_ss, preds_train_ss, title_suffix=f"ShuffleSplit_TRAIN{name}", save=True)
    # plot_confusion_matrices(y_test_ss, preds_test_ss, title_suffix=f"ShuffleSplit_TEST{name}", save=True)
    # plot_predictions_distribution(y_train_ss, preds_train_ss, title_suffix=f"ShuffleSplit_TRAIN{name}", save=True)
    # plot_predictions_distribution(y_test_ss, preds_test_ss, title_suffix=f"ShuffleSplit_TEST{name}", save=True)
    # print("\n")

    # # 3. LEAVEONE OUT (LOO)
    # print("--- Benchmark : Leave-One-Out (LOO) ---")
    # df_loo, preds_train_loo, y_train_loo, preds_test_loo, y_loo = run_loo(models, X, y)
    # print(df_loo.to_string())
    # df_loo.to_csv(os.path.join(output_dir, f"results_loo{name}.csv"))
    #
    # plot_confusion_matrices(y_train_loo, preds_train_loo, title_suffix=f"LOO_TRAIN{name}", save=True)
    # plot_confusion_matrices(y_loo, preds_test_loo, title_suffix=f"LOO_TEST{name}", save=True)




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
    train_metrics = get_metrics(y_train, y_train_pred)
    test_metrics = get_metrics(y_test, y_test_pred)

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