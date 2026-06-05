import os
from utils_MCLR import (
    load_and_preprocess_data,
    build_model_1,
    build_model_2,
    run_sampling,
    evaluate_and_save_results
)


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
    model_1 = build_model_1(df, dims, coords)

    print("\n=== MODEL 1 SAMPLING ===")
    idata_1 = run_sampling(model_1, draws=1000, tune=1000)

    variables_m1 = [
        "beta_age", "beta_gender", "beta_job", "beta_electric_bike", "beta_bike_use_frequency", "beta_bike_ownership",
        "beta_nbr_lanes", "beta_type", "beta_slope", "beta_speed", "beta_green"
    ]
    evaluate_and_save_results(model_1, idata_1, df, variables_m1, model_label="Model 1")

    # 4. Model 2 Execution (Restricted Model)
    print("\n=== MODEL 2 CONSTRUCTION ===")
    model_2 = build_model_2(df, dims, coords)

    print("\n=== MODEL 2 SAMPLING ===")
    idata_2 = run_sampling(model_2, draws=1000, tune=1000)

    variables_m2 = ["beta_nbr_lanes", "beta_type", "beta_slope", "beta_speed", "beta_green"]
    evaluate_and_save_results(model_2, idata_2, df, variables_m2, model_label="Model 2")


if __name__ == "__main__":
    main()