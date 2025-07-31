import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config.config import FEATURE_PREFIX, VALIDATION_SPLIT, SEED

def load_and_split_data(file_path="data/Augmented_Positive_Unlabeled_Data.csv"):
    df = pd.read_csv(file_path)
    
    # Split by labels
    D1_df = df[df["label"] == 1.0].reset_index(drop=True)
    Neg_df = df[df["label"] == 0.0].reset_index(drop=True)
    Unlabeled_df = df[df["label"].isna()].reset_index(drop=True)

    # Train/Val split from unlabeled
    unl_train_df, val_df = train_test_split(
        Unlabeled_df, test_size=VALIDATION_SPLIT, random_state=SEED, shuffle=True
    )

    return D1_df, Neg_df, unl_train_df, val_df


def normalize_features(D1_df, Neg_df, Unlabeled_df, val_df):
    feature_cols = [c for c in D1_df.columns if c.startswith(FEATURE_PREFIX)]
    
    # Extract raw features
    features_D1 = D1_df[feature_cols].values
    features_Neg = Neg_df[feature_cols].values
    features_Unl = Unlabeled_df[feature_cols].values
    features_Val = val_df[feature_cols].values

    # Normalize using D1
    scaler = StandardScaler()
    scaler.fit(features_D1)

    D1_norm = scaler.transform(features_D1)
    Neg_norm = scaler.transform(features_Neg)
    Unl_norm = scaler.transform(features_Unl)
    Val_norm = scaler.transform(features_Val)

    return D1_norm, Neg_norm, Unl_norm, Val_norm, feature_cols, scaler
