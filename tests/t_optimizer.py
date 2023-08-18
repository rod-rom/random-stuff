import numpy as np
from optim.optimizer import stochastic_gradient_descent
def test_stochastic_gradient_descent():
    # Generate a synthetic dataset
    X = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    y = np.array([2, 4, 6, 8, 10])

    # Train the model using SGD
    weights = stochastic_gradient_descent(X, y, learning_rate=0.01, num_epochs=100, batch_size=2)

    # Perform predictions
    y_pred = np.dot(X, weights)

    # Check the correctness of the predictions
    np.testing.assert_almost_equal(y_pred, y, decimal=2)