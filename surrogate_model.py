import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import joblib

def train_surrogate():
    # 1. Load the simulated "FEM" data
    try:
        data = pd.read_csv('simulated_fe_data.csv')
    except FileNotFoundError:
        print("Error: data file not found. Run data_simulator.py first.")
        return

    X = data[['relative_humidity_pct', 'temperature_c', 'mechanical_loading_kpa']]
    y = data['structural_stability']

    # 2. Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Train a Multi-layer Perceptron (Artificial Neural Network)
    # This acts as our "Surrogate Model"
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
    print("Training the Surrogate AI model (ANN) to mimic FEM results...")
    model.fit(X_train, y_train)

    # 4. Save results
    score = model.score(X_test, y_test)
    print(f"Model Accuracy (R^2): {score:.4f}")
    
    joblib.dump(model, 'surrogate_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Model and Scaler saved successfully.")

if __name__ == "__main__":
    train_surrogate()
