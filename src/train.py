import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score

def main():
    # 1. Connect the script to the light MLflow server running on your Windows host
    # WSL2 looks at the host machine via 127.0.0.1 or host.docker.internal
    #mlflow.set_tracking_uri("http://172.22.16.1:5000")
    mlflow.set_experiment("Credit_Card_Fraud_Detection")
    mlflow.set_tracking_uri("http://docker.internal")
    
    # Start an MLflow run to log everything automatically
    with mlflow.start_run(run_name="Hyperparameter_Tuning_Run"):

        # Mude o max_iter na linha 44 para testar um limite menor:
        max_iter = 50 

        
        # 2. Dataset path
        data_path = "data/raw/creditcard.csv"
        print(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path)
        
        # 3. Splitting features (X) and target (y) into raw numpy arrays for compatibility
        X = df.drop(columns=['Class']).values
        y = df['Class'].values
        
        # 4. Train/Test Split (80% train, 20% test)
        test_size = 0.2
        random_state = 42
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Log basic preprocessing parameters into MLflow
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        
        # 5. Applying RobustScaler
        print("Scaling features with RobustScaler...")
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 6. Training Model 1: Logistic Regression
        max_iter = 1000
        print("Training Logistic Regression model...")
        lr_model = LogisticRegression(max_iter=max_iter)
        lr_model.fit(X_train_scaled, y_train)
        
        # Evaluate Logistic Regression
        lr_preds = lr_model.predict(X_test_scaled)
        lr_recall = recall_score(y_test, lr_preds)
        lr_precision = precision_score(y_test, lr_preds)
        
        # Log Logistic Regression parameters and metrics
        mlflow.log_param("lr_max_iter", max_iter)
        mlflow.log_metric("lr_recall", lr_recall)
        mlflow.log_metric("lr_precision", lr_precision)
        
        # 7. Training Model 2: Random Forest
        print("Training Random Forest model (please wait)...")
        rf_model = RandomForestClassifier(random_state=random_state, n_jobs=-1)
        rf_model.fit(X_train_scaled, y_train)
        
        # Evaluate Random Forest
        rf_preds = rf_model.predict(X_test_scaled)
        rf_recall = recall_score(y_test, rf_preds)
        rf_precision = precision_score(y_test, rf_preds)
        
        # Log Random Forest metrics
        mlflow.log_metric("rf_recall", rf_recall)
        mlflow.log_metric("rf_precision", rf_precision)
        
        # 8. Saving artifacts into models/ directory locally for DVC tracking
        print("Saving models and scaler to 'models/' directory...")
        os.makedirs('models', exist_ok=True)
        joblib.dump(scaler, 'models/robust_scaler.joblib')
        joblib.dump(lr_model, 'models/logistic_regression_model.joblib')
        joblib.dump(rf_model, 'models/random_forest_model.joblib')
        
        # 9. Register the models inside MLflow artifacts tracking
        print("Registering models into MLflow artifacts tracking...")
        mlflow.sklearn.log_model(lr_model, "logistic_regression_model")
        mlflow.sklearn.log_model(rf_model, "random_forest_model")
        
        print("Success! All models trained, logged, and metrics sent to MLflow dashboard!")

if __name__ == "__main__":
    main()
