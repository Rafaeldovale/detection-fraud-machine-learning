import os
import joblib
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def main():
    data_path ="data/raw/creditcard.csv"
    print(f'Loding: {data_path}')
    df = pd.read_csv(data_path)


    X = df.drop(columns=['Class'])
    y = df['Class']

    # Train/test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print('Scaling feature with RobustScaler... ')
    scaler= RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training the model Logistic Regression... ")
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)

    print('Training the model Random Forest... ')
    rf_model = RandomForestClassifier(random_state=43, n_jobs=-1)
    rf_model.fit(X_train_scaled, y_train)

    print("\n========= Logistic Regression Performance =========")
    print(classification_report(y_test, lr_model.predict(X_test_scaled)))
    
    print("\n========= Random Forest Performance =========")
    print(classification_report(y_test, rf_model.predict(X_test_scaled)))


    print("Saving models and scaler to 'models/' directory...")
    os.makedirs('models', exist_ok=True)
    
    # Exporting artifacts using joblib
    joblib.dump(scaler, 'models/robust_scaler.joblib')
    joblib.dump(lr_model, 'models/logistic_regression_model.joblib')
    joblib.dump(rf_model, 'models/random_forest_model.joblib')
    
    print("✅ Success! All .joblib artifacts have been generated!")

if __name__ == "__main__":
    main()