import joblib
import numpy as np

print("🤖 Iniciando script de diagnóstico sem Uvicorn...")

# 1. Carregar artefatos
scaler = joblib.load("models/robust_scaler.joblib")
model = joblib.load("models/logistic_regression_model.joblib")

# 2. Criar um dado falso idêntico ao formato da linha 0 (vetor com 30 números)
# Uma transação fake com Time, V1...V28, Amount
dummy_input = np.zeros((1, 30)) 

try:
    print("\n🔹 Testando o Scaler...")
    X_scaled = scaler.transform(dummy_input)
    print(f"✅ Scaler funcionou! Formato da saída: {X_scaled.shape}")
    
    print("\n🔹 Testando o Predict...")
    prediction = model.predict(X_scaled)
    print(f"✅ Predict funcionou! Resposta: {prediction}")
    
    print("\n🔹 Testando o Predict Proba...")
    probabilities = model.predict_proba(X_scaled)
    print(f"✅ Predict Proba funcionou! Formato das probabilidades: {probabilities.shape}")
    print(f"➡️ Conteúdo das probabilidades: {probabilities}")
    
except Exception as e:
    import traceback
    print("\n🚨 EMBOSCADA DETECTADA! O erro aconteceu aqui:")
    print(traceback.format_exc())
