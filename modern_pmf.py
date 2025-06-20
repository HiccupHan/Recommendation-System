import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from main_model import probabilistic_model

class opt_pmf(probabilistic_model):
    def __init__(self, n_users, n_animes, lambda_U, lambda_V, uniq_users, uniq_animes, n_dimesions=50):
        self.n_users = n_users
        self.n_animes = n_animes
        self.lambda_U = lambda_U
        self.lambda_V = lambda_V
        self.n_dims = n_dimesions
        self.uniq_users = uniq_users
        self.uniq_animes = uniq_animes

        self.user_to_row = {user_id: i for i, user_id in enumerate(uniq_users)}
        self.anime_to_column = {anime_id: j for j, anime_id in enumerate(uniq_animes)}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.init_model()

    def init_model(self):
        class PMFModule(nn.Module):
            def __init__(self, n_users, n_items, n_dims):
                super().__init__()
                self.user_emb = nn.Embedding(n_users, n_dims)
                self.item_emb = nn.Embedding(n_items, n_dims)
                self.user_bias = nn.Embedding(n_users, 1)
                self.item_bias = nn.Embedding(n_items, 1)
                self.global_bias = nn.Parameter(torch.tensor(0.0))

                nn.init.normal_(self.user_emb.weight, 0, 0.1)
                nn.init.normal_(self.item_emb.weight, 0, 0.1)

            def forward(self, user_ids, item_ids):
                u = self.user_emb(user_ids)
                v = self.item_emb(item_ids)
                dot = (u * v).sum(dim=1)
                b_u = self.user_bias(user_ids).squeeze()
                b_i = self.item_bias(item_ids).squeeze()
                return dot + b_u + b_i + self.global_bias

        self.model = PMFModule(self.n_users, self.n_animes, self.n_dims).to(self.device)

    def train(self, train_set, test_set, n_epochs):
        train_ids, train_ratings = self._prepare_data(train_set)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        log_aps, rmse_train, rmse_test = [], [], []
        mae_train, mae_test = [], []

        for epoch in range(n_epochs):
            self.model.train()
            for batch in DataLoader(TensorDataset(train_ids, train_ratings), batch_size=2048, shuffle=True):
                user_batch, item_batch, rating_batch = batch[0][:, 0], batch[0][:, 1], batch[1]
                optimizer.zero_grad()
                preds = self.model(user_batch, item_batch)
                loss = criterion(preds, rating_batch)
                loss.backward()
                optimizer.step()
              
            rmse_tr, mae_tr = self.evaluate(train_set)
            rmse_te, mae_te = self.evaluate(test_set)
            rmse_train.append(rmse_tr)
            rmse_test.append(rmse_te)
            mae_train.append(mae_tr)
            mae_test.append(mae_te)
            if (epoch + 1) % 10 == 0 or epoch == 0:

                print(f"Epoch {epoch+1}: Train RMSE = {rmse_train[-1]:.4f}, Test RMSE = {rmse_test[-1]:.4f}")

        return log_aps, rmse_train, rmse_test, mae_train, mae_test

    def _prepare_data(self, df):
        user_ids = torch.tensor([self.user_to_row[u] for u in df['user_id']], dtype=torch.long)
        item_ids = torch.tensor([self.anime_to_column[i] for i in df['anime_id']], dtype=torch.long)
        ratings = torch.tensor(df['rating'].values, dtype=torch.float32)

        ids = torch.stack([user_ids, item_ids], dim=1).to(self.device)
        ratings = ratings.to(self.device)
        return ids, ratings

    def predict(self, user_id, anime_id):
        i = self.user_to_row.get(user_id, None)
        j = self.anime_to_column.get(anime_id, None)
        if i is None or j is None:
            return self.model.global_bias.item()  
        with torch.no_grad():
            uid = torch.tensor([i], dtype=torch.long).to(self.device)
            iid = torch.tensor([j], dtype=torch.long).to(self.device)
            pred = self.model(uid, iid).item()
        return np.clip(pred, 1.0, 10.0)

    def evaluate(self, dataset):
        y_true, y_pred = [], []
        for _, row in dataset.iterrows():
            y_true.append(row['rating'])
            y_pred.append(self.predict(row['user_id'], row['anime_id']))
        return root_mean_squared_error(y_true, y_pred), mean_absolute_error(y_true, y_pred)