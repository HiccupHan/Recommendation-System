import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from main_model import probabilistic_model
from sklearn.model_selection import KFold

class lda_model(probabilistic_model):
    def __init__(self, n_topics=10, max_iter=10, alpha=0.1, beta=0.1, reg_lambda=0.1):
        self.n_topics = n_topics
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.reg_lambda = reg_lambda
        self.vectorizer = CountVectorizer(
            stop_words='english',
            max_df=0.95,
            min_df=2,
            max_features=5000,
            ngram_range=(1, 2)
        )
        self.lda = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=max_iter,
            learning_method='batch',
            random_state=0,
            doc_topic_prior=alpha,
            topic_word_prior=beta
        )
        self.anime_topic_matrix = None
        self.user_profiles = dict()
        self.anime_ids = []
        self.user_ids = []
        self.global_bias = 0.0
        self.user_bias = {}
        self.anime_bias = {}
        self.rating_scale = 1.0
        self.anime_id_to_index = {}

    def train(self, train_set, test_set, n_epochs=None):
        print("in train, vectorizing anime dataset")
        train_set = train_set.dropna(subset=["description"])
        anime_text = train_set.drop_duplicates("anime_id")[["anime_id", "description"]]

        self.anime_ids = anime_text["anime_id"].tolist()
        self.user_ids = train_set["user_id"].unique().tolist()

        self.global_bias = train_set['rating'].mean()
        self.rating_scale = train_set['rating'].std()

        for user_id in self.user_ids:
            user_ratings = train_set[train_set["user_id"] == user_id]['rating']
            self.user_bias[user_id] = (user_ratings.mean() - self.global_bias) / (1 + self.reg_lambda)

        for anime_id in self.anime_ids:
            anime_ratings = train_set[train_set["anime_id"] == anime_id]['rating']
            if len(anime_ratings) > 0:
                self.anime_bias[anime_id] = (anime_ratings.mean() - self.global_bias) / (1 + self.reg_lambda)
            else:
                self.anime_bias[anime_id] = 0.0

        anime_desc_matrix = self.vectorizer.fit_transform(anime_text["description"])
        self.anime_topic_matrix = self.lda.fit_transform(anime_desc_matrix)
        self.anime_id_to_index = {aid: idx for idx, aid in enumerate(self.anime_ids)}

        print("building user topic profiles")
        for user_id in self.user_ids:
            user_data = train_set[train_set["user_id"] == user_id]
            topic_sum = np.zeros(self.n_topics)
            count = 0
            for _, row in user_data.iterrows():
                aid = row["anime_id"]
                rating = row["rating"]
                if aid in self.anime_id_to_index:
                    norm_rating = (rating - self.global_bias - self.user_bias[user_id] - self.anime_bias.get(aid, 0.0)) / self.rating_scale
                    topic_vec = self.anime_topic_matrix[self.anime_id_to_index[aid]]
                    topic_sum += topic_vec * norm_rating
                    count += 1
            self.user_profiles[user_id] = topic_sum / (count + self.reg_lambda) if count > 0 else np.zeros(self.n_topics)

    def predict(self, user_id, anime_id):
        if user_id not in self.user_profiles or anime_id not in self.anime_id_to_index:
            return self.global_bias

        user_profile = self.user_profiles[user_id]
        anime_topics = self.anime_topic_matrix[self.anime_id_to_index[anime_id]]
        
        base_pred = np.dot(user_profile, anime_topics) / (1 + self.reg_lambda)
        
        pred = (base_pred * self.rating_scale + 
                self.global_bias + 
                self.user_bias.get(user_id, 0.0) + 
                self.anime_bias.get(anime_id, 0.0))
        
        return np.clip(pred, 1.0, 10.0)

    def evaluate(self, dataset):
        y_true = []
        y_pred = []
        for _, row in dataset.iterrows():
            pred = self.predict(row["user_id"], row["anime_id"])
            y_true.append(row["rating"])
            y_pred.append(pred)
        return np.sqrt(mean_squared_error(y_true, y_pred)), mean_absolute_error(y_true, y_pred)

    def train_loss(self):
        return None

    def find_best_params(self, train_set, param_grid):
        best_rmse = float('inf')
        best_params = None
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        
        for n_topics in param_grid['n_topics']:
            for alpha in param_grid['alpha']:
                for beta in param_grid['beta']:
                    for reg_lambda in param_grid['reg_lambda']:
                        rmse_scores = []
                        
                        self.n_topics = n_topics
                        self.alpha = alpha
                        self.beta = beta
                        self.reg_lambda = reg_lambda
                        self.lda = LatentDirichletAllocation(
                            n_components=n_topics,
                            max_iter=self.max_iter,
                            learning_method='batch',
                            random_state=0,
                            doc_topic_prior=alpha,
                            topic_word_prior=beta
                        )
                        
                        for train_idx, val_idx in kf.split(train_set):
                            train_fold = train_set.iloc[train_idx]
                            val_fold = train_set.iloc[val_idx]
                            
                            self.train(train_fold, val_fold)
                            rmse, mae = self.evaluate(val_fold)
                            rmse_scores.append(rmse)
                        
                        avg_rmse = np.mean(rmse_scores)
                        print(f"Parameters: n_topics={n_topics}, alpha={alpha}, beta={beta}, reg_lambda={reg_lambda}, RMSE={avg_rmse:.4f}")
                        
                        if avg_rmse < best_rmse:
                            best_rmse = avg_rmse
                            best_params = {
                                'n_topics': n_topics,
                                'alpha': alpha,
                                'beta': beta,
                                'reg_lambda': reg_lambda
                            }
        
        return best_params
