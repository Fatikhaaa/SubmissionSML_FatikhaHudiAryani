import dagshub
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
import os
import shutil
import mlflow
import mlflow.sklearn

# Load dataset
train_path = "insurance_preprocessing/insurance_train_preprocessed.csv"
test_path = "insurance_preprocessing/insurance_test_preprocessed.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

X_train = train_df.drop(columns=["target"])
y_train = train_df["target"]
X_test = test_df.drop(columns=["target"])
y_test = test_df["target"]

# Jangan jalankan DAGsHub + MLflow lokal bersamaan, bisa running gantian atau dengan membuat file baru untuk membedakan keduanya agar tidak bentrok
# Lokal 
# mlflow.set_tracking_uri("http://127.0.0.1:5001/")

# DagsHub
dagshub.init(repo_owner='Fatikhaaa', repo_name='insurance-mlflow-fatikha', mlflow=True)
mlflow.set_experiment("Insurance_Cost_Prediction_Tuning")

# Hyperparameter space
n_estimators_range = np.linspace(50, 500, 5, dtype=int)   # 5 nilai antara 50-500
max_depth_range = np.linspace(5, 50, 5, dtype=int)        # 5 nilai antara 5-50

best_score = -np.inf
best_params = {}
best_model = None

for n_estimators in n_estimators_range:
    for max_depth in max_depth_range:
        with mlflow.start_run(run_name=f"rf_{n_estimators}_{max_depth}"):
            # Train
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)

            # Predict + metric
            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            rmse = root_mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)

            # Manual logging
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_metric("MSE", mse)
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("R2", r2)
            mlflow.log_metric("MAE", mae)    # tambahan 1
            mlflow.log_metric("MAPE", mape)  # tambahan 2

            # Save model terbaik
            if r2 > best_score:
                best_score = r2
                best_params = {"n_estimators": n_estimators, "max_depth": max_depth}
                best_model = model
                # --- Save model locally ---
                model_dir = "best_model"
                if os.path.exists(model_dir):
                    shutil.rmtree(model_dir)

                mlflow.sklearn.save_model(
                    sk_model=model,
                    path=model_dir
                )

                # --- Upload model ke DAGsHub ---
                mlflow.log_artifact(model_dir)

                # Tidak dihapus → biarkan tersimpan lokal
                print(f"[INFO] Best model updated → saved at: {model_dir}")
              

print("\n==================== FINAL RESULT ====================")
print("Best R2:", best_score)
print("Best Params:", best_params)
print("Model also uploaded to DAGsHub artifacts.")