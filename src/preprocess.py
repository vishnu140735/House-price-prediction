import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils import load_california_housing, train_test_split_df

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

def main():
    df = load_california_housing()
    X_train, X_test, y_train, y_test = train_test_split_df(df, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    _ = scaler.transform(X_test)  

    
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))

    
    pd.Series(X_train.columns).to_csv(os.path.join(MODEL_DIR, "feature_names.csv"), index=False, header=False)

    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    X_train.to_csv(os.path.join(data_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(data_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(data_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(data_dir, "y_test.csv"), index=False)

    print("Preprocessing complete. Scaler & feature names saved to models/. Data splits saved to data/.")

if __name__ == "__main__":
    main()
