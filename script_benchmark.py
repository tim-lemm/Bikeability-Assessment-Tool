from utils_benchmark import *

# Global Variables
random_state = 42
n_splits = 5
save = True
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
    df_tt, preds_train_tt, y_train_tt, preds_test_tt, y_test_tt = run_train_test_ML(
        models_ML, X, y, random_state=random_state
    )

    print("\n --> MCLR")
    # Retrieve the DataFrame AND prediction dictionaries (modes)
    df_results, test_probabilities, train_probabilities, preds_train_mclr, preds_test_mclr = run_train_test_MCLR(
        df, dims, random_state=random_state
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
        models_ML, X, y, n_splits=n_splits, random_state=random_state
    )

    print("\n --> MCLR")
    df_ss_mclr, preds_train_ss_mclr, _, preds_test_ss_mclr, _ = run_shufflesplit_MCLR(
        df, dims, n_splits=n_splits, random_state=random_state
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


def opti():
    X, y, _, _, _ = load_and_prepare_data(base_csv, personnes_csv, photos_csv)

    param_grid = {
        'learning_rate': [0.001, 0.01, 0.1, 1],
        'max_iter': [5,10,15,20, 50, 100, 200, 300],
        'max_leaf_nodes': [1,2,3,4,5,6,7,8,9,10],
        'max_depth': [1,2,3,5, 10, 31, 20],
        'l2_regularization': [0,0.1,0.2,0.5,1]
    }

    model = HistGradientBoostingClassifier(random_state=random_state)
    model_name = "HistGradientBoostingClassifier"

    # Run optimization
    best_model, best_parameter = optimize_model_hp(X, y, model, param_grid, scoring_metric="precision_weighted")
    df_best_parameters = pd.DataFrame([best_parameter])
    print(df_best_parameters.to_string())
    df_best_parameters.to_csv(os.path.join(output_dir, f"results_GS_best_parameter_{model_name}.csv"))

    # Train the best model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    best_model.fit(X_train, y_train)

    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    # Display results of the best model
    train_metrics = get_metrics_ML(y_train, y_train_pred)
    test_metrics = get_metrics_ML(y_test, y_test_pred)

    results = {"best_model": {}}
    for k, v in train_metrics.items():
        results["best_model"][f'Train {k}'] = v
    for k, v in test_metrics.items():
        results["best_model"][f'Test {k}'] = v

    df_results = pd.DataFrame(results).T
    df_results.to_csv(os.path.join(output_dir, f"results_GS_{model_name}.csv"))
    print(df_results.to_string())


if __name__ == "__main__":
    main()
    # opti()