import pandas as pd
import numpy as np

def generate_scientific_data(n_samples=1500):
    """
    Simulates a 'Digital Twin' training set based on:
    1. Hygrothermal Performance (Diffusion-Capillary Transport)
    2. Mechanical Characterization (Shear & Loading)
    """
    np.random.seed(42)
    
    # 1. Input Features (Variables they use)
    relative_humidity = np.random.uniform(30, 95, n_samples) # Vapor diffusion context
    temp_celsius = np.random.uniform(5, 45, n_samples)        # Thermal context
    mechanical_loading_kpa = np.random.uniform(50, 500, n_samples) # Garbowski's structural logic
    
    # 2. Scientific Degradation Formula (Non-linear surrogate)
    # Mimics Non-linear anisotropic diffusion and load-bearing capacity reduction
    humidity_factor = (relative_humidity / 100) ** 2  # Accelerated degradation at high RH
    thermal_factor = np.exp(0.02 * temp_celsius) / 2 # Temperature-accelerated diffusion
    
    # Predicted Variable: Effective Structural Stability (Surrogate target)
    stability = 1.0 - (0.4 * humidity_factor) - (0.1 * thermal_factor) - (0.3 * (mechanical_loading_kpa / 500))
    
    # Add Noise for Inverse Problem Context
    stability += np.random.normal(0, 0.03, n_samples)
    stability = np.clip(stability, 0, 1)
    
    # 3. Save for surrogate training
    df = pd.DataFrame({
        'relative_humidity_pct': relative_humidity,
        'temperature_c': temp_celsius,
        'mechanical_loading_kpa': mechanical_loading_kpa,
        'structural_stability': stability
    })
    
    df.to_csv('simulated_fe_data.csv', index=False)
    print(f"Generated {n_samples} samples based on Hygrothermal & Mechanical Logic.")

if __name__ == "__main__":
    generate_scientific_data()
