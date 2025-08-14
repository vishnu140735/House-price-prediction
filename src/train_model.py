import os
import argparse
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from utils import load_california_housing, train_test_split_df

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

def get_data_scaled():
    df = load_california_housing()
    X_train, X_test, y_train, y_test = train_test_split_df(df, test_size=0.2, random_state=42)

    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        scaler = StandardScaler().fit(X_train)
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(scaler, scaler_path)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values

def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds) 
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)
    return model, {"MAE": mae, "RMSE": rmse, "R2": r2}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["linear", "ridge", "lasso", "tree", "rf"], default="linear")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge/Lasso regularization strength")
    parser.add_argument("--max_depth", type=int, default=None, help="Decision tree max depth")
    parser.add_argument("--n_estimators", type=int, default=200, help="RandomForest n_estimators")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = get_data_scaled()

    if args.model == "linear":
        model = LinearRegression()
    elif args.model == "ridge":
        model = Ridge(alpha=args.alpha, random_state=42)
    elif args.model == "lasso":
        model = Lasso(alpha=args.alpha, random_state=42, max_iter=10000)
    elif args.model == "tree":
        model = DecisionTreeRegressor(max_depth=args.max_depth, random_state=42)
    elif args.model == "rf":
        model = RandomForestRegressor(n_estimators=args.n_estimators, random_state=42)
    else:
        raise ValueError("Unsupported model")

    model, metrics = train_and_evaluate(model, X_train, X_test, y_train, y_test)


    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"{args.model}.joblib")
    joblib.dump(model, model_path)

    print("Model trained and saved to:", model_path)
    print("Metrics:", metrics)

if __name__ == "__main__":
    main()
