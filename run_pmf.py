import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from pmf import pmf

print("start")
anime_df = pd.read_csv('data/anime.csv')
anime_df = anime_df.rename(columns={'rating': 'avg_rating'})
rating_df = pd.read_csv('data/rating.csv')

print("read complete")

rating_df = rating_df.sample(n=100_000, random_state=42)

rating_df = rating_df[rating_df['rating'] != -1]
# rating_df['rating'] = (rating_df['rating'] - 1) / 9.0 # Normalize ratings to [0, 1]
combined_df = pd.merge(anime_df, rating_df, on='anime_id', how='inner')

print("merge complete")

uniq_users = combined_df['user_id'].unique()
uniq_animes = combined_df['anime_id'].unique()

n_users = len(uniq_users)
n_animes = len(uniq_animes)

df_copy = combined_df.copy()


train_set = df_copy.sample(frac=0.75, random_state=0)
test_set = df_copy.drop(train_set.index)

print("init model")
pmf_model = pmf(n_users=n_users, n_animes=n_animes, lambda_U=0.3, lambda_V=0.3, uniq_users=uniq_users, uniq_animes=uniq_animes, n_dimesions=20)

print("start train")
log_ps, rmse_train, rmse_test = pmf_model.train(train_set=train_set, test_set=test_set,n_epochs=150)
print('RMSE of training set:', pmf_model.evaluate(train_set))
print('RMSE of testing set:', pmf_model.evaluate(test_set))