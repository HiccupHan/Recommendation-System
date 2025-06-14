import pandas as pd
from sklearn.model_selection import train_test_split
from lda import lda_model

print("loading data")
anime_df = pd.read_csv('data/anime.csv')
rating_df = pd.read_csv('data/rating.csv')
rating_df = rating_df.sample(n=100_000, random_state=42)
rating_df = rating_df[rating_df['rating'] != -1]

anime_df = anime_df.rename(columns={'rating': 'avg_rating'})
anime_df = anime_df.rename(columns={'genre': 'description'})

combined_df = pd.merge(anime_df, rating_df, on='anime_id', how='inner')
combined_df = combined_df.dropna(subset=["description"])


train_set = combined_df.sample(frac=0.75, random_state=0)
test_set = combined_df.drop(train_set.index)

print("training")
lda_recommender = lda_model(n_topics=20, max_iter=10)
lda_recommender.train(train_set=train_set, test_set=test_set)

print("eval")
rmse_train = lda_recommender.evaluate(train_set)
rmse_test = lda_recommender.evaluate(test_set)
print('RMSE Train:', rmse_train)
print('RMSE Test:', rmse_test)
