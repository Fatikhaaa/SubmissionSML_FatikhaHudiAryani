import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

# Set MLflow Tracking (gunakan local atau remote)
mlflow.set_tracking_uri("http://127.0.0.1:5001/")
# Set experiment name
mlflow.set_experiment("Insurance_Cost_Prediction")

# Load dataset
train_path =  "insurance_preprocessing/insurance_train_preprocessed.csv"
test_path = "insurance_preprocessing/insurance_test_preprocessed.csv"

train_df = pd.read_csv(train_path)
test_df =  pd.read_csv(test_path)

X_train = train_df.drop(columns=["target"])
y_train = train_df["target"]
X_test = test_df.drop(columns=["target"])
y_test = test_df["target"]

# Training dan logging otomatis
with mlflow.start_run(run_name="manual_run"):
    # Aktifkan autolog
    mlflow.sklearn.autolog()

    # Train model
    model = RandomForestRegressor()
    model.fit(X_train,y_train)

    # Evaluasi model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )

    print("✅ Model trained and autologged successfully.")
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")
    