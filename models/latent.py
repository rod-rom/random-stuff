import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def latent_factor_model(R, num_factors, learning_rate=0.01, num_epochs=100, batch_size=32, regularization=0.01):
    num_users, num_items = R.shape
    
    # Initialize user and item latent factor matrices randomly
    P = np.random.randn(num_users, num_factors)
    Q = np.random.randn(num_items, num_factors)

    for epoch in range(num_epochs):
        # Shuffle the data
        permutation = np.random.permutation(num_users)
        R_shuffled = R[permutation]

        for batch_start in range(0, num_users, batch_size):
            batch_end = min(batch_start + batch_size, num_users)
            batch_R = R_shuffled[batch_start:batch_end]

            # Compute gradient and update user and item latent factor matrices
            for user_idx, item_idx, rating in np.ndenumerate(batch_R):
                error = rating - np.dot(P[user_idx], Q[item_idx])
                P[user_idx] += learning_rate * (error * Q[item_idx] - regularization * P[user_idx])
                Q[item_idx] += learning_rate * (error * P[user_idx] - regularization * Q[item_idx])

    return P, Q

def matrix_factorization(R, num_factors):
    # Perform singular value decomposition (SVD)
    U, sigma, Vt = np.linalg.svd(R, full_matrices=False)

    # Truncate the matrices to the desired number of factors
    U_truncated = U[:, :num_factors]
    sigma_truncated = np.diag(sigma[:num_factors])
    Vt_truncated = Vt[:num_factors, :]

    # Compute the factorized matrix
    R_pred = U_truncated @ sigma_truncated @ Vt_truncated

    return R_pred

def select_num_factors(R, num_factors_list, test_size=0.2, random_state=42):
    train_data, test_data = train_test_split(R, test_size=test_size, random_state=random_state)
    train_data = np.array(train_data)
    test_data = np.array(test_data)

    best_num_factors = None
    best_rmse = np.inf

    for num_factors in num_factors_list:
        R_pred = matrix_factorization(R, num_factors)
        rmse = np.sqrt(mean_squared_error(test_data, R_pred))

        if rmse < best_rmse:
            best_rmse = rmse
            best_num_factors = num_factors

    return best_num_factors, best_rmse