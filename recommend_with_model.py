import pandas as pd
import numpy as np
from keras.models import Sequential, load_model
from typing import Any


def model_recommender(model: Sequential, user_id: Any, how_many: int=10) -> None:
    ratings_df = pd.read_csv('ratings.csv')
    movies_df = pd.read_csv('movies.csv')

    # create one dictionary mapping userId to userId count in the dataframe
    user_ids = ratings_df["userId"].unique().tolist()
    userId_to_usercount_dict = {x: i for i, x in enumerate(user_ids)}

    # create one dictionary mapping movieId to movieId count in the dataframe and the other does the opposite
    movie_ids = ratings_df["movieId"].unique().tolist()
    movieId_to_moviecount_dict = {x: i for i, x in enumerate(movie_ids)}
    moviecount_to_movieId_dict = {i: x for i, x in enumerate(movie_ids)}

    # create features of the mapped users and movies
    ratings_df["user"] = ratings_df["userId"].map(userId_to_usercount_dict)
    ratings_df["movie"] = ratings_df["movieId"].map(movieId_to_moviecount_dict)

    # choose user_id randomly or use a user_id specified by input, must be a valid user_id
    if user_id in ('Random', 'random', 'Rand', 'rand'):
        user_id = ratings_df.userId.sample(1).iloc[0]

    elif isinstance(user_id, int):    # if input is int and not invalid - do nothing and don't change input
        if user_id not in user_ids:
            raise ValueError('Invalid user ID! please try another one')

    else:   # if input is not an int or 'random' its invalid for sure
        raise ValueError('user_id input must be "random" or an integer!')

    movies_watched_by_user = ratings_df[ratings_df.userId == user_id]

    # get movies not watched by the user and turn to list which contains the movies not watched by user
    movies_not_watched = movies_df[~movies_df["movieId"].isin(movies_watched_by_user.movieId.values)]["movieId"]
    movies_not_watched = list(set(movies_not_watched).intersection(set(movieId_to_moviecount_dict.keys())))

    # create list of the corresponding movie ID's
    movies_not_watched = [[movieId_to_moviecount_dict.get(x)] for x in movies_not_watched]

    # do the same with the users
    user_encoder = userId_to_usercount_dict.get(user_id)

    '''
    Create an ndarray of shape (movies_not_watched_by_user, 2). Meaning it has two "columns". for each row:
    Left column has our chosen user's ID.
    Right column has ID's of movies the user didn't watch.

    This way we have an array which is in the correct shape to put in the model and will predict all unseen movies of
    our user. At the end we cast the values to be int64 to overcome a ValueError.
    '''
    user_movie_array = np.hstack(([[user_encoder]] * len(movies_not_watched), movies_not_watched)).astype('int64')

    # predict on all movies and get the movies with the highest predicted rating (amount is given as input to function)
    ratings = model.predict(user_movie_array).flatten()
    top_ratings_indices = ratings.argsort()[-how_many:][::-1]

    # get the movies to recommend from the index encoding dictionary
    recommended_movie_ids = [moviecount_to_movieId_dict.get(movies_not_watched[x][0]) for x in top_ratings_indices]

    print("Showing recommendations for user: {}".format(user_id))
    print("====" * 9)
    print("Movies with high ratings from user")
    print("----" * 8)
    top_movies_user = (movies_watched_by_user.sort_values(by="rating", ascending=False)
        .head(5)
        .movieId.values
    )

    # get rows from movies_df of movies which were highly rated by the user and print
    movies_df_rows = movies_df[movies_df["movieId"].isin(top_movies_user)]
    for row in movies_df_rows.itertuples():
        print(row.title, ":", row.genres)

    print("----" * 8)
    print("Top 10 movie recommendations")
    print("----" * 8)

    # get rows from movies_df of movies which will be recommended to the user
    recommended_movies = movies_df[movies_df["movieId"].isin(recommended_movie_ids)]
    for row in recommended_movies.itertuples():
        print(row.title, ":", row.genres)

    return


# can be 'final_cb_model' or 'final_mf_model'
MODEL = load_model('final_mf_model')


def main():
    model_recommender(MODEL, 'random')


if __name__ == "__main__":
    main()

