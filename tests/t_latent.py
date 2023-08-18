import numpy as np
from models.latent import latent_factor_model


def test_latent_factor_model():
    # Create a sample utility matrix
    R = np.array([[5, 3, 0, 1],
                  [4, 0, 0, 1],
                  [1, 1, 0, 5],
                  [1, 0, 0, 4],
                  [0, 1, 5, 4]])

    num_factors = 2
    learning_rate = 0.01
    num_epochs = 100
    batch_size = 2
    regularization = 0.01

    # Train the latent factor model
    P, Q = latent_factor_model(R, num_factors, learning_rate, num_epochs, batch_size, regularization)

    # Check the shape of the learned latent factor matrices
    assert P.shape == (R.shape[0], num_factors)
    assert Q.shape == (R.shape[1], num_factors)

    # Make predictions
    R_pred = np.dot(P, Q.T)

    # Check the correctness of the predictions
    for user_idx in range(R.shape[0]):
        for item_idx in range(R.shape[1]):
            rating = R[user_idx, item_idx]
            predicted_rating = R_pred[user_idx, item_idx]
            assert np.isclose(predicted_rating, rating, rtol=1e-3)