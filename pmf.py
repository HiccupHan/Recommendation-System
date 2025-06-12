import numpy as np
from sklearn.metrics import root_mean_squared_error
from main_model import probabilistic_model

class pmf(probabilistic_model):
    def __init__(self, n_users, n_animes, lambda_U, lambda_V, uniq_users, uniq_animes, n_dimesions=50):
        self.n_dims = n_dimesions
        self.n_users = n_users
        self.n_animes = n_animes
        self.U = np.random.normal(0.0, 1.0 / lambda_U, (self.n_dims, self.n_users))
        self.V = np.random.normal(0.0, 1.0 / lambda_V, (self.n_dims, self.n_animes))
        self.lambda_U = lambda_U
        self.lambda_V = lambda_V
        self.R = np.zeros((n_users, n_animes))
        self.user_to_row = {user_id: i for i, user_id in enumerate(uniq_users)}
        self.anime_to_column = {anime_id: j for j, anime_id in enumerate(uniq_animes)}

    def update_parameters(self):
        for i in range(self.n_users):
            item_indices = self.R[i, :] > 0
            if np.sum(item_indices) == 0:
                continue
            V_j = self.V[:, item_indices]
            ratings_i = self.R[i, item_indices]
            A = V_j @ V_j.T + self.lambda_U * np.identity(self.n_dims)
            b = V_j @ ratings_i.T
            self.U[:, i] = np.linalg.solve(A, b)

        for j in range(self.n_animes):
            user_indices = self.R[:, j] > 0
            if np.sum(user_indices) == 0:
                continue
            U_i = self.U[:, user_indices]
            ratings_j = self.R[user_indices, j]
            A = U_i @ U_i.T + self.lambda_V * np.identity(self.n_dims)
            b = U_i @ ratings_j.T
            self.V[:, j] = np.linalg.solve(A, b)

    def predict(self, user_id, anime_id):
        i = self.user_to_row[user_id]
        j = self.anime_to_column[anime_id]
        pred = self.U[:, i].T @ self.V[:, j]
        pred = np.clip(pred, 0.0, 1.0)  # since training target is in [0, 1]
        return pred

    def evaluate(self, dataset):
        y_true = []
        y_pred = []
        for _, row in dataset.iterrows():
            pred = self.predict(row['user_id'], row['anime_id']) * 9.0 + 1.0  # now safe
            y_true.append(row['rating'] * 9.0 + 1.0)
            y_pred.append(pred)
        return root_mean_squared_error(y_true, y_pred)

    def log_a_posteriori(self):
        error = 0.0
        for i in range(self.n_users):
            for j in range(self.n_animes):
                if self.R[i, j] > 0:
                    pred = self.U[:, i].T @ self.V[:, j]
                    error += (self.R[i, j] - pred) ** 2
        return -0.5 * (
            error +
            self.lambda_U * np.sum(self.U ** 2) +
            self.lambda_V * np.sum(self.V ** 2)
        )

    def train_loss(self):
        error = 0.0
        count = 0
        for i in range(self.n_users):
            for j in range(self.n_animes):
                if self.R[i, j] > 0:
                    pred = self.U[:, i].T @ self.V[:, j]
                    error += (self.R[i, j] - pred) ** 2
                    count += 1
        return error / count if count > 0 else 0.0

    def train(self, train_set, test_set, n_epochs):
        for _, row in train_set.iterrows():
            i = self.user_to_row[row.user_id]
            j = self.anime_to_column[row.anime_id]
            self.R[i, j] = row.rating

        log_aps = []
        rmse_train = [self.evaluate(train_set)]
        rmse_test = [self.evaluate(test_set)]

        for k in range(n_epochs):
            self.update_parameters()
            log_ap = self.log_a_posteriori()
            log_aps.append(log_ap)

            rmse_train.append(self.evaluate(train_set))
            rmse_test.append(self.evaluate(test_set))

            if (k + 1) % 10 == 0:
                print('Log p a-posteriori at iteration', k + 1, ':', log_ap,
                      ", RMSE train:", rmse_train[-1],
                      ", RMSE test:", rmse_test[-1],
                      ", Train loss:", self.train_loss())

        preds = [self.predict(row['user_id'], row['anime_id']) * 9 + 1 for _, row in test_set.iterrows()]
        print("Predicted rating range:", min(preds), "to", max(preds))

        return log_aps, rmse_train, rmse_test
