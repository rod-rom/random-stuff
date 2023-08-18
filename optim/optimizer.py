import numpy as np


def stochastic_gradient_descent(X, y, learning_rate=0.01, num_epochs=100, batch_size=32):
    num_samples, num_features = X.shape
    num_batches = num_samples // batch_size

    # Initialize random weights
    weights = np.random.randn(num_features)

    for epoch in range(num_epochs):
        # Shuffle the data
        permutation = np.random.permutation(num_samples)
        X_shuffled = X[permutation]
        y_shuffled = y[permutation]

        for batch in range(num_batches):
            # Select mini-batch
            start = batch * batch_size
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            # Compute gradient
            gradient = np.zeros(num_features)
            for i in range(batch_size):
                prediction = np.dot(X_batch[i], weights)
                error = prediction - y_batch[i]
                gradient += error * X_batch[i]

            # Update weights
            weights -= learning_rate * gradient / batch_size

    return weights