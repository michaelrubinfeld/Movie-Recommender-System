import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras import optimizers
from keras.layers import Embedding
from keras.optimizers import Adam, Adamax
from tensorflow_addons.optimizers import LazyAdam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import combinations
import matplotlib.pyplot as plt
import random
from gc import collect
from typing import Any, Sequence


SPLIT_RATE = 0.8    # 80% of the data for train 20% for test. 20% of the remaining train data will go to validation
EPOCHS = 300


def extract_title(title: str) -> str:
    year = title[len(title) - 5:len(title) - 1]

    # deal with title without the year in the title
    if year.isnumeric():
        title_no_year = title[:len(title) - 7]
        return title_no_year
    else:
        return title


def extract_year(title: str) -> Any:
    year = title[len(title) - 5:len(title) - 1]

    # deal with title without the year in the title
    if year.isnumeric():
        return int(year)
    else:
        return np.nan


ratings_df = pd.read_csv('ratings.csv')
movies_df = pd.read_csv('movies.csv')

# sort by rating date
ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'])
ratings_df = ratings_df.sort_values(by=['timestamp'])

# change the column name from title to title_year
movies_df.rename(columns={'title': 'title_year'}, inplace=True)

# remove leading and ending whitespaces in title_year
movies_df['title_year'] = movies_df['title_year'].apply(lambda x: x.strip())

# create the columns for title and year
movies_df['title'] = movies_df['title_year'].apply(extract_title)
movies_df['year'] = movies_df['title_year'].apply(extract_year)

# change 'Sci-Fi' to 'SciFi' and 'Film-Noir' to 'Noir' (all genres without all letters in their names)
movies_df['genres'] = movies_df['genres'].str.replace('Sci-Fi', 'SciFi')
movies_df['genres'] = movies_df['genres'].str.replace('Film-Noir', 'Noir')

# create one dictionary mapping userId to userId count in the dataframe
user_ids = ratings_df["userId"].unique().tolist()
userId_to_usercount_dict = {x: i for i, x in enumerate(user_ids)}

# create one dictionary mapping movieId to movieId count in the dataframe
movie_ids = ratings_df["movieId"].unique().tolist()
movieId_to_moviecount_dict = {x: i for i, x in enumerate(movie_ids)}

# create features of the mapped users and movies
ratings_df["user"] = ratings_df["userId"].map(userId_to_usercount_dict)
ratings_df["movie"] = ratings_df["movieId"].map(movieId_to_moviecount_dict)

# Split to data and label
X = ratings_df[["user", "movie"]].values
y = ratings_df["rating"].values

# split to train, validation and test
train_val_indices = int(SPLIT_RATE * X.shape[0])
X_train_val, X_test, y_train_val, y_test = (
    X[:train_val_indices],
    X[train_val_indices:],
    y[:train_val_indices],
    y[train_val_indices:],
)

train_indices = int(SPLIT_RATE * X_train_val.shape[0])
X_train, X_val, y_train, y_val = (
    X_train_val[:train_indices],
    X_train_val[train_indices:],
    y_train_val[:train_indices],
    y_train_val[train_indices:],
)

# get amount of unique users
num_users = len(userId_to_usercount_dict)

print(f'train has {X_train.shape[0]} ratings, validation has {X_val.shape[0]} ratings, and test has {X_test.shape[0]}.')

# will be used as global bias
train_global_avg = np.mean(list(y_train))

'''
In the following part, we're following the same ideas of handling bias initialization as was taught in class:
1. Create a train average vector (ndarray).

2. Initialize a matrix with the matching bias' dimensions. The matrix's values are sampled from the he_normal 
distribution (mean: 0, std: sqrt(2/fan_in)), to match the distribution initialized in the user embedding layer.
** fan-in --> number of input units in the weight tensor 

3. Finally, for each user, we add to it's bias the difference between the user's average and the global bias 
'''


def get_user_bias_init_matrix() -> np.ndarray:
    mode_ratings_df = ratings_df[['userId', 'rating']]
    train_avg = mode_ratings_df.groupby('userId', as_index=False).mean()
    train_avg = train_avg['rating'].to_numpy()

    user_biases_init_to_avgs = np.random.normal(0, (2 / (num_users + 1)) ** 0.5, size=(num_users + 1, 1))
    user_biases_init_to_avgs[:num_users, 0] += train_avg - train_global_avg

    return user_biases_init_to_avgs


class ContentBasedModel(keras.Model):
    def __init__(self, init_biases_to_avgs: bool, movie_genres_avgs: Sequence, embedding_matrix: np.array, n_users: int,
                 vocab_len: int, embedding_dim: int, global_bias: Any, **kwargs):
        super(ContentBasedModel, self).__init__(**kwargs)
        self.n_users = n_users
        self.user_biases_init_to_avgs = get_user_bias_init_matrix()
        self.global_bias = tf.convert_to_tensor(value=global_bias, dtype=tf.float32)
        self.user_embedding = Embedding(
            num_users + 1,
            embedding_dim,
            embeddings_initializer='he_normal',
            mask_zero=True
        )

        self.movie_embedding = Embedding(
            vocab_len,
            embedding_dim,
            weights=[embedding_matrix],
            trainable=False,
            mask_zero=True
        )

        if init_biases_to_avgs:
            self.user_bias = Embedding(num_users + 1, 1, weights=[self.user_biases_init_to_avgs])
            self.movie_bias = Embedding(vocab_len, 1, weights=[movie_genres_avgs])
        else:
            self.user_bias = Embedding(num_users + 1, 1)
            self.movie_bias = Embedding(vocab_len, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = tf.squeeze(self.user_bias(inputs[:, 0]))

        movie_vector = self.movie_embedding(inputs[:, 1])
        movie_bias = tf.squeeze(self.movie_bias(inputs[:, 1]))

        global_bias = self.global_bias
        dot_user_movie = tf.tensordot(user_vector, movie_vector, 2)

        # Add all the components (including biases)
        return dot_user_movie + user_bias + movie_bias + global_bias


# dictionary to track the results and ultimately return the best hyperparameters
tuning_results_tracker = {}
lr_const_init = 1e-4    # lr will be reduced during the training process automatically when getting to a plateau


def build_model(init_biases_to_avgs: bool, max_genres_ngram_range: int, optimizer: optimizers,
                batch_size: int, is_hyperparameter_tuning: bool) -> Sequential:
    '''
    We would like to take into account every combination of genres in all movies so we will calculate the tf-idf of
    genre combinations in each movie present in the dataframe.

    This method is prone to overfitting since using a combination of multiple genres per movie is very specific.
    For this reason, we will tune the maximal amount of movies in a combination (ngram range of genres) as a
    hyperparameter. For example, if ngram range is 3, genre combinations of 4 and above will not be considered.
    '''

    tf_idf_vector = TfidfVectorizer(analyzer=lambda genres: (combination for combination_range_size in
                                                             range(1, max_genres_ngram_range + 1) for combination in
                                                             combinations(genres.split('|'),
                                                                          r=combination_range_size)),
                                    stop_words='english')

    # create the tfidf matrix matrix and get its ndarray dense representation (inportant)
    tfidf_genre_combinations_matrix = tf_idf_vector.fit_transform(movies_df['genres'].values.astype(str))
    tfidf_genre_combinations_matrix = tfidf_genre_combinations_matrix.toarray()

    # get the amount of movies (vocab length)
    vocab_len_tfidf = tfidf_genre_combinations_matrix.shape[0]

    # get the amount of genres combinations to represent the movies (essentially the embedding dimension)
    embedding_dim_tfidf = tfidf_genre_combinations_matrix.shape[1]

    '''
    In order to fully utilize on the tf-idf embedding layer concept, we will keep the embeddings given by the
    TfidfVectorizer as they are. The tf-idf embedding matrix has a given embedding dimension, If we change
    it when transforming the tf-idf matrix to an embedding layer, we will miss the whole point of using the tf-idf
    matrix inplace of learning the movie's embedding vectors. In addition, the user's embedding dimension should match
    the movie's. Therefore, the model's embedding dimension will always be inferred from the tf-idf matrix and 
    will not be tuned as a hyperparameter.
    '''

    # get each genre combination's averages to initialize the movie bias layer's weights
    tfidf_movie_genre_avgs = np.mean(tfidf_genre_combinations_matrix, axis=1).reshape(vocab_len_tfidf, 1)

    # instantiate the content based model
    cb_model = ContentBasedModel(init_biases_to_avgs=init_biases_to_avgs, movie_genres_avgs=tfidf_movie_genre_avgs,
                                 embedding_matrix=tfidf_genre_combinations_matrix, n_users=num_users,
                                 vocab_len=vocab_len_tfidf, embedding_dim=embedding_dim_tfidf,
                                 global_bias=train_global_avg)

    cb_model.compile(loss='mean_squared_error', optimizer=optimizer(lr=lr_const_init))

    # CALLBACKS:
    # if the val_loss score does not improve within X epochs - stop training
    es = EarlyStopping(monitor='val_loss', verbose=1, patience=5)

    # if the val_loss does not improve in X epochs - divide the learning rate by 2. Stop dividing if reached 1e-8.
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-8, verbose=1)

    history = cb_model.fit(
        x=X_train, y=y_train,
        steps_per_epoch=X_train.shape[0] // batch_size,
        validation_steps=X_val.shape[0] // batch_size,
        batch_size=batch_size, callbacks=[es, reduce_lr],
        epochs=EPOCHS, verbose=1, validation_data=(X_val, y_val),
    )

    final_loss_score = history.history['val_loss'][-1]

    if is_hyperparameter_tuning:
        optimizer_name = str(optimizer).split('.')[-1][:-2]
        curr_hyperparameters = f'init_biases_to_avgs: {init_biases_to_avgs}, ' \
                               f'max_genres_ngram_range: {max_genres_ngram_range}, optimizer: {optimizer_name}, ' \
                               f'batch_size: {batch_size}'

        print(f'curr_try hyperparameters: {curr_hyperparameters}')

        print(f'MODEL LOSS: {final_loss_score}\n')
        tuning_results_tracker[curr_hyperparameters] = final_loss_score

    else:
        print(f'FINAL MODEL LOSS TO BE SAVED: {final_loss_score}')

        # save model
        cb_model.save('final_cb_model')

    _ = collect()
    return history


# set to False to train a regular network
is_hyperparameter_tuning = False


def main():
    if is_hyperparameter_tuning:
        n_tries = 10

        ### params grid ###
        is_init_biases_to_avgs = [True, False]
        max_genres_ngram_ranges = [1, 2, 3]
        # LazyAdam == SparseAdam
        # Adamax -> version of Adam sometimes superior to adam, especially in models with embeddings
        optimizers = [LazyAdam, Adam, Adamax]
        batch_sizes = [16, 32, 64, 128]

        for curr_try in range(n_tries):
            print(f'currunt try: {curr_try + 1}/{n_tries}')

            # randomly choose hyperparameters and train a model using them
            init_biases_to_avgs = random.choice(is_init_biases_to_avgs)
            max_genres_ngram_range = random.choice(max_genres_ngram_ranges)
            optimizer = random.choice(optimizers)
            batch_size = random.choice(batch_sizes)

            build_model(init_biases_to_avgs=init_biases_to_avgs, max_genres_ngram_range=max_genres_ngram_range,
                        optimizer=optimizer, batch_size=batch_size, is_hyperparameter_tuning=is_hyperparameter_tuning)

        best_hyperparameters = min(zip(tuning_results_tracker.values(), tuning_results_tracker.keys()))[1]
        print(f'THE BEST HYPERPARAMETERS ARE: {best_hyperparameters}')

    else:
        # Hyperparameters are tuned, let's train the final model on the train set and save it's epoch
        # with the best loss on the validation set
        best_init_biases_to_avgs = False
        best_max_genres_ngram_range = 3
        best_optimizer = Adam
        best_batch_size = 32

        history = build_model(init_biases_to_avgs=best_init_biases_to_avgs,
                              max_genres_ngram_range=best_max_genres_ngram_range, optimizer=best_optimizer,
                              batch_size=best_batch_size, is_hyperparameter_tuning=is_hyperparameter_tuning)

        # plot results
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.grid(linestyle='--', linewidth=0.5)
        plt.title("Content based Model Loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["Train", "Val"], loc="upper left")
        plt.show()


if __name__ == "__main__":
    main()

