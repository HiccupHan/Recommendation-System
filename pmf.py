import numpy as np
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from main_model import probabilistic_model

class pmf(probabilistic_model):
    def __init__(self, n_users, n_animes, lambda_U, lambda_V, uniq_users, uniq_animes, n_dimesions=50, bias_reg=5.0):
        self.n_dims = n_dimesions
        self.n_users = n_users
        self.n_animes = n_animes
        self.lambda_U = lambda_U
        self.lambda_V = lambda_V
        self.bias_reg = bias_reg

        self.U = np.random.normal(0.0, 0.1, (self.n_dims, self.n_users))
        self.V = np.random.normal(0.0, 0.1, (self.n_dims, self.n_animes))
        self.R = np.full((n_users, n_animes), np.nan)

        self.user_to_row = {user_id: i for i, user_id in enumerate(uniq_users)}
        self.anime_to_column = {anime_id: j for j, anime_id in enumerate(uniq_animes)}

        self.b_u = np.zeros(self.n_users)
        self.b_i = np.zeros(self.n_animes)
        self.global_bias = 0.0

        self.rating_mask = None
        self.user_indices = []
        self.item_indices = []

    def update_parameters(self):
        for i in self.user_indices:
            item_indices = self.rating_mask[i]
            V_j = self.V[:, item_indices]
            ratings_i = self.R[i, item_indices] - self.b_i[item_indices] - self.global_bias
            A = V_j @ V_j.T + self.lambda_U * np.identity(self.n_dims)
            b = V_j @ ratings_i.T
            self.U[:, i] = np.linalg.solve(A, b)
            pred_ratings = self.U[:, i].T @ V_j
            self.b_u[i] = np.sum(ratings_i - pred_ratings) / (len(ratings_i) + self.bias_reg)

        for j in self.item_indices:
            user_indices = self.rating_mask[:, j]
            U_i = self.U[:, user_indices]
            ratings_j = self.R[user_indices, j] - self.b_u[user_indices] - self.global_bias
            A = U_i @ U_i.T + self.lambda_V * np.identity(self.n_dims)
            b = U_i @ ratings_j.T
            self.V[:, j] = np.linalg.solve(A, b)
            pred_ratings = self.V[:, j].T @ U_i
            self.b_i[j] = np.sum(ratings_j - pred_ratings) / (len(ratings_j) + self.bias_reg)

    def predict(self, user_id, anime_id):
        i = self.user_to_row.get(user_id)
        j = self.anime_to_column.get(anime_id)
        if i is None or j is None:
            return self.global_bias  # fallback
        pred = self.U[:, i].T @ self.V[:, j] + self.b_u[i] + self.b_i[j] + self.global_bias
        return np.clip(pred, 1.0, 10.0)

    def evaluate(self, dataset):
        y_true = []
        y_pred = []
        for _, row in dataset.iterrows():
            pred = self.predict(row['user_id'], row['anime_id'])
            y_true.append(row['rating'])
            y_pred.append(pred)
        return root_mean_squared_error(y_true, y_pred), mean_absolute_error(y_true, y_pred)

    def log_a_posteriori(self):
        error = 0.0
        for i in self.user_indices:
            for j in self.item_indices:
                if self.rating_mask[i, j]:
                    pred = self.U[:, i].T @ self.V[:, j] + self.b_u[i] + self.b_i[j] + self.global_bias
                    error += (self.R[i, j] - pred) ** 2
        return -0.5 * (
            error +
            self.lambda_U * np.sum(self.U ** 2) +
            self.lambda_V * np.sum(self.V ** 2)
        )

    def train(self, train_set, test_set, n_epochs):
        for _, row in train_set.iterrows():
            i = self.user_to_row.get(row.user_id)
            j = self.anime_to_column.get(row.anime_id)
            if i is not None and j is not None:
                self.R[i, j] = row.rating

        self.global_bias = np.nanmean(self.R)
        self.rating_mask = ~np.isnan(self.R)
        self.user_indices = np.where(self.rating_mask.sum(axis=1) > 0)[0]
        self.item_indices = np.where(self.rating_mask.sum(axis=0) > 0)[0]

        log_aps = []
        rmse_train = []
        rmse_test = []
        mae_train = []
        mae_test = []
        
        for k in range(n_epochs):
            self.update_parameters()
            log_ap = self.log_a_posteriori()
            log_aps.append(log_ap)
            rmse_tr, mae_tr = self.evaluate(train_set)
            rmse_te, mae_te = self.evaluate(test_set)
            rmse_train.append(rmse_tr)
            rmse_test.append(rmse_te)
            mae_train.append(mae_tr)
            mae_test.append(mae_te)
            if (k + 1) % 10 == 0 or k == 0 or k == n_epochs - 1:
                print('Log p a-posteriori at iteration', k + 1, ':', log_ap,
                      ", RMSE train:", rmse_train[-1],
                      ", RMSE test:", rmse_test[-1],
                      ", MAE train:", mae_train[-1],
                      ", MAE test:", mae_test[-1])

        return log_aps, rmse_train, rmse_test, mae_train, mae_test
