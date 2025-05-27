import numpy as np
from sklearn.metrics import mean_squared_error
from main_model import probabilistic_model

class pmf(probabilistic_model):
    def __init__(self, n_users, n_animes, lambda_U, lambda_V, uniq_users, uniq_animes):
        self.n_dims = 10
        self.n_users = n_users
        self.n_animes = n_animes
        self.U = np.random.normal(0.0, 1.0 / lambda_U, (self.n_dims, self.n_users))
        self.V = np.random.normal(0.0, 1.0 / lambda_V, (self.n_dims, self.n_animes))
        self.lambda_U = lambda_U
        self.lambda_V = lambda_V
        self.R = np.zeros((n_users, n_animes))
        self.user_to_row = {}
        self.anime_to_column = {}
        for i, user_id in enumerate(uniq_users):
            self.user_to_row[user_id] = i
        for j, anime_id in enumerate(uniq_animes):
            self.anime_to_column[anime_id] = j
        self.max_rating = 0
        self.min_rating = 0
        
    def update_parameters(self):
        # Update U (user latent matrix)
        for i in range(self.n_users):
            item_indices = self.R[i, :] > 0
            if np.sum(item_indices) == 0:
                continue  # skip users with no ratings

            V_j = self.V[:, item_indices]
            ratings_i = self.R[i, item_indices]

            A = V_j @ V_j.T + self.lambda_U * np.identity(self.n_dims)
            b = V_j @ ratings_i.T
            self.U[:, i] = np.linalg.solve(A, b)

        # Update V (item latent matrix)
        for j in range(self.n_animes):
            user_indices = self.R[:, j] > 0
            if np.sum(user_indices) == 0:
                continue  # skip items with no ratings

            U_i = self.U[:, user_indices]
            ratings_j = self.R[user_indices, j]

            A = U_i @ U_i.T + self.lambda_V * np.identity(self.n_dims)
            b = U_i @ ratings_j.T
            self.V[:, j] = np.linalg.solve(A, b)
                

    def log_a_posteriori(self):        
        UV = np.dot(self.U.T, self.V)
        R_UV = (self.R[self.R > 0] - UV[self.R > 0])
        
        return -0.5 * (np.sum(R_UV ** 2) + self.lambda_U * np.sum(np.dot(self.U, self.U.T)) + self.lambda_V * np.sum(np.dot(self.V, self.V.T)))
        

    def predict(self, user_id, anime_id):
        r_ij = self.U[:, self.user_to_row[user_id]].T.reshape(1, -1) @ self.V[:, self.anime_to_column[anime_id]].reshape(-1, 1)

        return 0 if self.max_rating == self.min_rating else ((r_ij[0][0] - self.min_rating) / (self.max_rating - self.min_rating)) * 5.0



    def evaluate(self, dataset):
        ground_truths = []
        predictions = []
        
        for index, row in dataset.iterrows():
            ground_truths.append(row.loc['rating'])
            predictions.append(self.predict(row.loc['user_id'], row.loc['anime_id']))
        
        return mean_squared_error(ground_truths, predictions, squared=False)
        


    def update_max_min_ratings(self):
        predictions = self.U.T @ self.V
        self.min_rating = np.min(predictions)
        self.max_rating = np.max(predictions)
        
    def train(self, train_set, test_set, n_epochs):
        for index, row in train_set.iterrows():
            i = self.user_to_row[row.user_id]
            j = self.anime_to_column[row.anime_id]
            self.R[i, j] = row.rating
        
        log_aps = []
        rmse_train = []
        rmse_test = []

        self.update_max_min_ratings()
        rmse_train.append(self.evaluate(train_set))
        rmse_test.append(self.evaluate(test_set))
        
        for k in range(n_epochs):
            self.update_parameters()
            log_ap = self.log_a_posteriori()
            log_aps.append(log_ap)

            if (k + 1) % 10 == 0:
                self.update_max_min_ratings()

                rmse_train.append(self.evaluate(train_set))
                rmse_test.append(self.evaluate(test_set))
                print('Log p a-posteriori at iteration', k + 1, ':', log_ap)

        self.update_max_min_ratings()

        return log_aps, rmse_train, rmse_test