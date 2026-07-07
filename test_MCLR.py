import os
from utils_MCLR import (
    load_and_preprocess_data,
    build_model_1,
    build_model_2,
    run_sampling,
    evaluate_and_save_results,
    prior_predictive_checks,
    run_benchmark
)
import matplotlib.pyplot as plt
import arviz as az


def main():
    # Ensure outputs directory exists
    os.makedirs("outputs/model_results/MCLR", exist_ok=True)

    # 1. Paths definitions
    base_csv = "data/survey/data-base.csv"
    personnes_csv = "data/survey/data-personnes.csv"
    photos_csv = "data/survey/data-photos.csv"

    # 2. Preprocessing
    print("\n=== DATA PREPROCESSING ===")
    df, dims, coords = load_and_preprocess_data(base_csv, personnes_csv, photos_csv)

    # 3. Model 1 Execution (Full Model)
    print("\n=== MODEL 1 CONSTRUCTION ===")
    model_1 = build_model_1(df, dims)

    print("\n=== MODEL 1 SAMPLING ===")
    idata_1 = run_sampling(model_1, draws=1000, tune=1000)

    variables_m1 = [
        "beta_age", "beta_gender", "beta_job", "beta_electric_bike", "beta_bike_use_frequency", "beta_bike_ownership",
        "beta_nbr_lanes", "beta_type", "beta_slope", "beta_speed", "beta_green"
    ]
    evaluate_and_save_results(model_1, idata_1, df, variables_m1, model_label="Model 1")

    prior_predictive_checks(model_1, save=True, model_label="Model 1")
    # 4. Model 2 Execution (Restricted Model)
    print("\n=== MODEL 2 CONSTRUCTION ===")
    model_2 = build_model_2(df, dims)

    print("\n=== MODEL 2 SAMPLING ===")
    idata_2 = run_sampling(model_2, draws=1000, tune=1000)

    variables_m2 = ["beta_nbr_lanes", "beta_type", "beta_slope", "beta_speed", "beta_green"]
    evaluate_and_save_results(model_2, idata_2, df, variables_m2, model_label="Model 2")

    prior_predictive_checks(model_2, save=True, model_label="Model 2")

    # 5. Model comparison (In-sample information criteria)
    model_dict = {
        "Model 1": idata_1,
        "Model 2": idata_2
    }
    df_compare = az.compare(model_dict, var_name="y_obs")
    print("\n=== COMPARISON RESULTS (LOO / WAIC) ===")
    print(df_compare)
    az.plot_compare(df_compare)
    plt.savefig("outputs/model_results/MCLR/model_comparison_loo.png")
    plt.show()

    # 6. Benchmark Validation (Out-of-sample performance via 80/20 Train-Test split)
    print("\n=== CROSS-VALIDATION BENCHMARK ===")
    model_factories = {
        "Model 1": build_model_1,
        "Model 2": build_model_2
    }

    df_benchmark = run_benchmark(
        df=df,
        dims=dims,
        coords=coords,
        model_factories=model_factories,
        n_splits=5,
        draws=1000,
        tune=1000
    )

    print("\n=== FINAL BENCHMARK PERFORMANCE REPORT ===")
    print(df_benchmark.to_string(index=False))

    # Sauvegarde du rapport final en fichier Excel
    df_benchmark.to_excel("outputs/model_results/MCLR/benchmark_validation_results.xlsx", index=False)
    print(
        "\nLe rapport de benchmark a été enregistré sous 'outputs/model_results/MCLR/benchmark_validation_results.xlsx'.")


if __name__ == "__main__":
    main()