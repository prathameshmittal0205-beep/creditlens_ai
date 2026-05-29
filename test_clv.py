import pandas as pd
from utils.preprocessing import process_uploaded_data, engineer_features, prepare_lifetimes_data
from models.clv import train_clv_models

try:
    df = pd.read_csv("realistic_fintech_dataset.csv")
    features = engineer_features(df)
    summary_df = prepare_lifetimes_data(df)
    print("Summary length:", len(summary_df))

    bgf, ggf = train_clv_models(summary_df)
    if bgf is None:
        print("Models failed to train.")
    else:
        print("Models trained successfully!")
except Exception as e:
    print("Training failed with error:", e)
