from pycaret.classification import setup, compare_models
from pycaret.datasets import get_data
import joblib

iris = get_data('iris')

setup(data=iris, target='species', session_id=42,
      verbose=False, html=False)

best_model = compare_models()
joblib.dump(best_model, 'iris_model.pkl')

print("Model saved successfully!")
