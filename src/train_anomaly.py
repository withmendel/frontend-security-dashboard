"""Train an anomaly detector on simulated front-end user behavior logs."""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from pathlib import Path

RANDOM_STATE = 42

def generate_synthetic_logs(n_samples=1000):
    """Simulate user interaction logs: [click_speed_ms, form_fill_time_s, mouse_movements]"""
    np.random.seed(RANDOM_STATE)
    # Normal users: slower, more mouse movements
    normal = np.random.normal(loc=[500, 30, 150], scale=[100, 5, 30], size=(int(n_samples*0.9), 3))
    # Bots/Attackers: very fast, few mouse movements
    anomalies = np.random.normal(loc=[50, 2, 5], scale=[10, 1, 2], size=(int(n_samples*0.1), 3))
    
    X = np.vstack([normal, anomalies])
    # Ensure no negative values
    X = np.abs(X)
    return X

def train_model():
    X = generate_synthetic_logs()
    print(f"Training on {X.shape[0]} samples...")
    
    # Isolation Forest is perfect for "security" because it finds outliers
    model = IsolationForest(contamination=0.1, random_state=RANDOM_STATE)
    model.fit(X)
    
    # Save the model
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    joblib.dump(model, model_dir / "anomaly_model.pkl")
    print("Model saved to models/anomaly_model.pkl")

if __name__ == "__main__":
    train_model()
