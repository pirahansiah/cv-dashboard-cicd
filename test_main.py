import numpy as np
import joblib
from sklearn.metrics import mean_squared_error

def test_model():
    # Load the model
    model = joblib.load('model.joblib')

    # Generate some sample test data
    X_test = np.array([[i] for i in range(101, 111)])
    y_test = np.array([2 * i + 1 for i in range(101, 111)])

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f"Test Mean Squared Error: {mse}")

    # Assert that the model performance is as expected
    assert mse < 1e-10, "Model Mean Squared Error is too high!"
