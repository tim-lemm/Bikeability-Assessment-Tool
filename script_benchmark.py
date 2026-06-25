from utils_benchmark import run_train_test, run_shufflesplit, run_loo, load_and_prepare_data, get_models, plot_confusion_matrices
import pandas as pd
import os

random_state=42
n_splits=5

def main():
    base_csv = "data/survey/data-base.csv"
    personnes_csv = "data/survey/data-personnes.csv"
    photos_csv = "data/survey/data-photos.csv"

    X, y = load_and_prepare_data(base_csv, personnes_csv, photos_csv)
    models = get_models()

    output_dir = "outputs/model_results/benchmark"
    os.makedirs(output_dir, exist_ok=True)

    # 1. TRAIN / TEST SPLIT
    print("--- Benchmark : Train / Test Split ---")
    df_tt, preds_train_tt, y_train_tt, preds_test_tt, y_test_tt = run_train_test(models, X, y, random_state=random_state)
    print(df_tt.to_string())
    df_tt.to_csv(os.path.join(output_dir, "results_train_test.csv"))

    plot_confusion_matrices(y_train_tt, preds_train_tt, title_suffix="TrainTest_TRAIN", save=True)
    plot_confusion_matrices(y_test_tt, preds_test_tt, title_suffix="TrainTest_TEST", save=True)
    print("\n")

    # 2. SHUFFLE SPLIT
    print(f"--- Benchmark : ShuffleSplit ({n_splits}-fold) ---")
    df_ss, preds_train_ss, y_train_ss, preds_test_ss, y_test_ss = run_shufflesplit(models, X, y, n_splits=n_splits, random_state=random_state)
    print(df_ss.to_string())
    df_ss.to_csv(os.path.join(output_dir, "results_shufflesplit.csv"))

    plot_confusion_matrices(y_train_ss, preds_train_ss, title_suffix="ShuffleSplit_TRAIN", save=True)
    plot_confusion_matrices(y_test_ss, preds_test_ss, title_suffix="ShuffleSplit_TEST", save=True)
    print("\n")

    # 3. LEAVEONE OUT (LOO)
    print("--- Benchmark : Leave-One-Out (LOO) ---")
    df_loo, preds_train_loo, y_train_loo, preds_test_loo, y_loo = run_loo(models, X, y)
    print(df_loo.to_string())
    df_loo.to_csv(os.path.join(output_dir, "results_loo.csv"))

    plot_confusion_matrices(y_train_loo, preds_train_loo, title_suffix="LOO_TRAIN", save=True)
    plot_confusion_matrices(y_loo, preds_test_loo, title_suffix="LOO_TEST", save=True)

if __name__ == "__main__":
    main()