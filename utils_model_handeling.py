import numpy as np
import joblib
from pyexpat import model
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

def load_model(model_name, filepath):
    return joblib.load(f"{filepath}/{model_name}.joblib")

def convert_df(edge_df):
    edge_df_for_predicition = edge_df.copy()
    edge_df_for_predicition = edge_df_for_predicition.rename(columns={"type_bike": "type",
                                                                      "speed_car": "speed",
                                                                      "green_overlap_percentage": "green",
                                                                      "nbr_car_lane": "nbr_lane"})

    type_mapping = {"Pas d'aménagement cyclable" : 0, "Pas d\'aménagement cyclable":0,
                    "Chaussée partagée aménagée" : 1,
                    "Aménagement séparé partagé" : 2,
                    "Aménagement séparé non partagé" : 3}

    edge_df_for_predicition["nom_amenagement"] = edge_df_for_predicition["type"].map(type_mapping)

    edge_df_for_predicition["speed"] = pd.cut(edge_df_for_predicition["speed"],
                                              bins = [-np.inf,1,10,20,np.inf],
                                              labels = [0,1,2,3]).fillna(0).astype(int)

    edge_df_for_predicition["slope"] = pd.cut(edge_df_for_predicition["slope"],
                                              bins = [-np.inf,2,6,np.inf],
                                              labels = [0,1,2]).astype(int)

    edge_df_for_predicition["green"] = pd.cut(edge_df_for_predicition["green"],
                                              bins=[-np.inf, 15, 75, np.inf],
                                              labels=[0, 1, 2]).astype(int)

    edge_df_for_predicition["nbr_lane"] = pd.cut(edge_df_for_predicition["nbr_lane"],
                                              bins=[-np.inf, 1, 2, 3, np.inf],
                                              labels=[0, 1, 2, 3]).fillna(0).astype(int)

    return edge_df_for_predicition

def preprocess_data(edge_df):
    cat_cols = ["type", "speed", "green", "nbr_lane", "slope"]
    enc_cols = ["type_0", "type_1", "type_2", "type_3", "speed_0", "speed_1", "speed_2", "speed_3","green_0", "green_1", "green_2", "nbr_lane_0", "nbr_lane_1", "nbr_lane_2","nbr_lane_3", "slope_0", "slope_1", "slope_2"]
    edge_df_for_predicition = convert_df(edge_df)
    enc = OneHotEncoder(dtype=np.int64, drop=None)
    enc_output = enc.fit_transform(edge_df_for_predicition[cat_cols]).toarray()
    encoded_cols = enc.get_feature_names_out(cat_cols)
    edge_df_for_predicition = pd.DataFrame(enc_output, columns=encoded_cols, index=edge_df_for_predicition.index)
    for col in enc_cols:
        if col not in edge_df_for_predicition.columns:
            edge_df_for_predicition[col]=0

    edge_df_for_predicition[["age_1", "age_2", "age_3", "bike_use_frequency_0","bike_use_frequency_1","bike_use_frequency_2","bike_use_frequency_3","bike_use_frequency_4","electric_bike_True","electric_bike_False","gender_0","gender_1","gender_2","bike_ownership_0", "bike_ownership_1"]] = edge_df[["age_1", "age_2", "age_3", "bike_use_frequency_0","bike_use_frequency_1","bike_use_frequency_2","bike_use_frequency_3","bike_use_frequency_4","electric_bike_True","electric_bike_False","gender_0","gender_1","gender_2","bike_ownership_0", "bike_ownership_1"]]
    # liste_cat = [f"nbr_lane_{i}" for i in range(4)] + [f"speed_{i}" for i in range(4)] + \
    #             [f"slope_{i}" for i in range(3)] + [f"green_{i}" for i in range(3)] + \
    #             [f"type_{i}" for i in range(4)]

    liste_cat = [f"nbr_lane_{i}" for i in range(4)] + [f"speed_{i}" for i in range(4)] + \
                [f"slope_{i}" for i in range(3)] + [f"green_{i}" for i in range(3)] + \
                [f"type_{i}" for i in range(4)] + ["age_1", "age_2", "age_3", "bike_use_frequency_0","bike_use_frequency_1","bike_use_frequency_2","bike_use_frequency_3","bike_use_frequency_4","electric_bike_True","electric_bike_False","gender_0","gender_1","gender_2","bike_ownership_0", "bike_ownership_1"]
    X = edge_df_for_predicition[liste_cat].copy()
    return X

def predict_notes(edge_df, model):

    edge_df_for_predicition = preprocess_data(edge_df)
    features_ordre = model.feature_names_in_
    edge_df_reordered = edge_df_for_predicition[features_ordre]
    prediction = model.predict(edge_df_reordered)
    return prediction

def apply_bi_model(edge_df, model):
    edge_df["note"] = predict_notes(edge_df, model)
    edge_df["note"] = edge_df["note"] + 1
    return edge_df


