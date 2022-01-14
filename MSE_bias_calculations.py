import pandas as pd
import numpy as np
import pickle

SPLIT_RATE = 0.8

# get ratings data and sort it by rating timestamp for the splits later
ratings_df = pd.read_csv('ratings.csv')
ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'])
ratings_df = ratings_df.sort_values(by=['timestamp'])

train_val_indices = int(SPLIT_RATE * ratings_df.shape[0])
train_val_ratings_df, test_ratings_df = (
    ratings_df[:train_val_indices],
    ratings_df[train_val_indices:]
)

train_indices = int(SPLIT_RATE * train_val_ratings_df.shape[0])
train_ratings_df, val_ratings_df = (
    train_val_ratings_df[:train_indices],
    train_val_ratings_df[train_indices:]
)

print(f'train has {train_ratings_df.shape[0]} ratings')
print(f'validation has {val_ratings_df.shape[0]} ratings')
print(f'test has {test_ratings_df.shape[0]} ratings\n')


def ratings_df_to_dict(df):
    ratings_dict = {}
    for index, row in df.iterrows():
        ratings_dict[(row.movieId, row.userId)] = int(row.rating)
    return ratings_dict

# Turn dataframes to dicts to ease calculation complexity later
train_set = ratings_df_to_dict(train_ratings_df)
val_set = ratings_df_to_dict(val_ratings_df)
test_set = ratings_df_to_dict(test_ratings_df)

# GLOBAL BIAS
train_global_avg = np.mean(list(train_set.values()))

# Get rating averages per user and per movie for all users and movies in the dataset
user_ratings_df = ratings_df[['userId', 'rating']]
train_user_avg = user_ratings_df.groupby('userId', as_index=False).mean()
train_user_avg = train_user_avg['rating'].tolist()

movie_ratings_df = ratings_df[['movieId', 'rating']]
train_movie_avg = movie_ratings_df.groupby('movieId', as_index=False).mean()
train_movie_avg = train_movie_avg['rating'].tolist()

# Val User MSE
val_user_mse = np.mean([(r - (train_user_avg[u] if u < len(train_user_avg) else train_global_avg)) ** 2
                        for (m, u), r in val_set.items()])

# Val Movie MSE
val_movie_mse = np.mean([(r - (train_movie_avg[m] if m < len(train_movie_avg) else train_global_avg)) ** 2
                        for (m, u), r in val_set.items()])

# Val User-Movie MSE
val_user_movie_mse = np.mean([(r - (train_user_avg[u] + train_movie_avg[m] - train_global_avg if
                                    u < len(train_user_avg) and m < len(train_movie_avg) else train_global_avg)) ** 2
                              for (m, u), r in val_set.items()])

# Test User MSE
test_user_mse = np.mean([(r - (train_user_avg[u] if u < len(train_user_avg) else train_global_avg)) ** 2
                        for (m, u), r in test_set.items()])

# Test Movie MSE
test_movie_mse = np.mean([(r - (train_movie_avg[m] if m < len(train_movie_avg) else train_global_avg)) ** 2
                          for (m, u), r in test_set.items()])

# Test User-Movie MSE
test_user_movie_mse = np.mean([(r - (train_user_avg[u] + train_movie_avg[m] - train_global_avg if
                                     u < len(train_user_avg) and m < len(train_movie_avg) else train_global_avg)) ** 2
                               for (m, u), r in test_set.items()])

print(f'GLOBAL BIAS: {train_global_avg}')
print(f'Val User MSE: {val_user_mse}')
print(f'Val Movie MSE: {val_movie_mse}')
print(f'Val User-Movie MSE: {val_user_movie_mse}')
print(f'test User MSE: {test_user_mse}')
print(f'test Movie MSE: {test_movie_mse}')
print(f'test User-Movie MSE: {test_user_movie_mse}')


output_dict = {'global_bias': train_global_avg,
               'val_user_mse': val_user_mse,
               'val_movie_mse': val_movie_mse,
               'val_user_movie_mse': val_user_movie_mse,
               'test_user_mse': test_user_mse,
               'test_movie_mse': test_movie_mse,
               'test_user_movie_mse': test_user_movie_mse
               }

# with open('mse_bias_claculations.pickle', 'wb') as handle:
#     pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


'''
USE THE CODE BELOW TO LOAD THE DATA IN THE JUPYTER NOTEBOOK
'''
# with open('mse_bias_claculations.pickle', 'rb') as handle:
#     mse_bias_claculations_dict = pickle.load(handle)