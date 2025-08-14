import os
import argparse
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils import load_california_housing, train_test_split_df, plot_pred_vs_actual, plot_residuals

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["linear", "ridge", "lasso", "tree", "rf"], default="linear")
    args = parser.parse_args()


    df = load_california_housing()
    X_train, X_test, y_train, y_test = train_test_split_df(df, test_size=0.2, random_state=42)

    
    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError("scaler.joblib not found. Run preprocess.py first.")
    scaler = joblib.load(scaler_path)
    X_test_scaled = scaler.transform(X_test)

    
    model_path = os.path.join(MODEL_DIR, f"{args.model}.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Train the model first.")
    model = joblib.load(model_path)

    
    preds = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)  
    rmse = np.sqrt(mse)                    
    r2 = r2_score(y_test, preds)

    
    pva_path = plot_pred_vs_actual(y_test.values, preds, filename=f"predicted_vs_actual_{args.model}.png")
    res_path = plot_residuals(y_test.values, preds, filename=f"residual_plot_{args.model}.png")

    print("Evaluation complete.")
    print({"MAE": mae, "RMSE": rmse, "R2": r2})
    print("Saved plots:")
    print(pva_path)
    print(res_path)

if __name__ == "__main__":
    main()
