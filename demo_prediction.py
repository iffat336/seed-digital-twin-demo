import joblib
import numpy as np
import time

def run_digital_twin_prediction(moisture, pressure):
    """
    Simulates a real-time Digital Twin response.
    Takes live sensor data and gives an instant prediction using the ML Surrogate.
    """
    # 1. Load our trained AI "Brain"
    try:
        model = joblib.load('surrogate_model.pkl')
        scaler = joblib.load('scaler.pkl')
    except FileNotFoundError:
        print("Error: Model files not found. Run surrogate_model.py first.")
        return

    # 2. Take live "Sensor Data"
    sensor_input = np.array([[moisture, pressure]])
    sensor_input_scaled = scaler.transform(sensor_input)

    # 3. Predict Instantly
    start_time = time.time()
    prediction = model.predict(sensor_input_scaled)[0]
    end_time = time.time()

    print("-" * 30)
    print(f"Digital Twin Response (Live Monitoring):")
    print(f"Inputs: Moisture={moisture}%, Pressure={pressure}kPa")
    print(f"Predicted Seed Structural Integrity: {prediction:.2%}")
    print(f"Time taken to predict: {(end_time - start_time)*1000:.4f} ms")
    
    if prediction < 0.5:
        print("WARNING: High risk of structural failure (Seed Damage)!")
    else:
        print("Status: Seed structure is stable.")
    print("-" * 30)

if __name__ == "__main__":
    # Example: Run a prediction for a seed in a high-moisture, high-pressure environment
    run_digital_twin_prediction(moisture=22.5, pressure=150.0)
    
    # Example: Run a prediction for a seed in a safe environment
    run_digital_twin_prediction(moisture=8.0, pressure=30.0)
