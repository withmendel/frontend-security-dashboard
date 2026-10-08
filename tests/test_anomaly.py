import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from train_anomaly import generate_synthetic_logs

def test_data_shape():
    X = generate_synthetic_logs(100)
    assert X.shape == (100, 3)

def test_no_negative_values():
    X = generate_synthetic_logs(100)
    assert (X >= 0).all()
