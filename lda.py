import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error
from main_model import probabilistic_model

class lda_model(probabilistic_model):
    def __init__(self, n_topics=10, max_iter=10):
        self.n_topics = n_topics
        self.max_iter = max_iter
        self.vectorizer = CountVectorizer(stop_words='english', max_df=0.95, min_df=2)
        self.lda = LatentDirichletAllocation(n_components=n_topics, max_iter=max_iter, learning_method='batch', random_state=0)
        self.anime_topic_matrix = None
        self.user_profiles = dict()
        self.anime_ids = []
        self.user_ids = []

    def train(self, train_set, test_set, n_epochs=None):
        print("in train, vectorizing anime dataset")
        train_set = train_set.dropna(subset=["description"])
        anime_text = train_set.drop_duplicates("anime_id")[["anime_id", "description"]]

        self.anime_ids = anime_text["anime_id"].tolist()
        self.user_ids = train_set["user_id"].unique().tolist()

        anime_desc_matrix = self.vectorizer.fit_transform(anime_text["description"])
        self.anime_topic_matrix = self.lda.fit_transform(anime_desc_matrix)

        self.anime_id_to_index = {aid: idx for idx, aid in enumerate(self.anime_ids)}
        self.user_id_to_index = {uid: idx for idx, uid in enumerate(self.user_ids)}

        print("build user topic profilcs")
        for user_id in self.user_ids:
            user_data = train_set[train_set["user_id"] == user_id]
            topic_sum = np.zeros(self.n_topics)
            count = 0
            for _, row in user_data.iterrows():
                aid = row["anime_id"]
                rating = row["rating"]
                if aid in self.anime_id_to_index:
                    topic_vec = self.anime_topic_matrix[self.anime_id_to_index[aid]]
                    topic_sum += topic_vec * rating
                    count += rating
            self.user_profiles[user_id] = topic_sum / count if count > 0 else np.zeros(self.n_topics)

    def predict(self, user_id, anime_id):
        if user_id not in self.user_profiles or anime_id not in self.anime_id_to_index:
            return 5.0
        user_profile = self.user_profiles[user_id]
        anime_topics = self.anime_topic_matrix[self.anime_id_to_index[anime_id]]
        return np.clip(np.dot(user_profile, anime_topics) * 10, 1.0, 10.0)

    def evaluate(self, dataset):
        y_true = []
        y_pred = []
        for _, row in dataset.iterrows():
            pred = self.predict(row["user_id"], row["anime_id"])
            y_true.append(row["rating"])
            y_pred.append(pred)
        return np.sqrt(mean_squared_error(y_true, y_pred))

    def train_loss(self):
        # cuz lda is unsuperevised
        return None
