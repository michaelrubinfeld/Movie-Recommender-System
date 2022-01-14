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
import matplotlib.pyplot as plt
import random
from gc import collect
from typing import Any

SPLIT_RATE = 0.8    # 80% of the data for train 20% for test. 20% of the remaining train data will go to validation
EPOCHS = 300

# get ratings data and sort it by rating timestamp for the splits later
ratings_df = pd.read_csv('ratings.csv')
ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'])
ratings_df = ratings_df.sort_values(by=['timestamp'])

# create one dictionary mapping userId to userId count in the dataframe
user_ids = ratings_df["userId"].unique().tolist()
userId_to_usercount_dict = {x: i for i, x in enumerate(user_ids)}

# create one dictionary mapping movieId to movieId count in the dataframe and the other does the opposite
movie_ids = ratings_df["movieId"].unique().tolist()
movieId_to_moviecount_dict = {x: i for i, x in enumerate(movie_ids)}

# create features of the mapped users and movies
ratings_df["user"] = ratings_df["userId"].map(userId_to_usercount_dict)
ratings_df["movie"] = ratings_df["movieId"].map(movieId_to_moviecount_dict)

# get amount of unique movies and users
num_users = len(userId_to_usercount_dict)
num_movies = len(movieId_to_moviecount_dict)

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

print(f'train has {X_train.shape[0]} ratings, validation has {X_val.shape[0]} kratings, and test has {X_test.shape[0]}.')

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


def get_bias_init_matrix(mode: str) -> np.ndarray:
    if mode != 'userId' and mode != 'movieId':
        raise ValueError('Must be a valid column from the ratings dataframe')
    mode_ratings_df = ratings_df[[mode, 'rating']]
    train_avg = mode_ratings_df.groupby(mode, as_index=False).mean()
    train_avg = train_avg['rating'].to_numpy()

    if mode == 'userId':
        n = num_users
    else:
        n = num_movies
    biases_init_to_avgs = np.random.normal(0, (2 / (n + 1)) ** 0.5, size=(n + 1, 1))
    biases_init_to_avgs[:n, 0] += train_avg - train_global_avg

    return biases_init_to_avgs



class MatrixFactorizationNet(keras.Model):
    def __init__(self, init_biases_to_avgs: bool, n_users: int, n_movies: int,
                 embedding_dim: int, global_bias: Any, **kwargs):
        super(MatrixFactorizationNet, self).__init__(**kwargs)
        self.num_users = n_users
        self.num_movies = n_movies
        self.user_biases_init_to_avgs = get_bias_init_matrix(mode='userId')
        self.movie_biases_init_to_avgs = get_bias_init_matrix(mode='movieId')
        self.global_bias = tf.convert_to_tensor(value=global_bias, dtype=tf.float32)
        self.user_embedding = Embedding(
            n_users + 1,
            embedding_dim,
            embeddings_initializer='he_normal',
            mask_zero=True     # use 0 as padding
        )

        self.movie_embedding = Embedding(
            n_movies + 1,
            embedding_dim,
            embeddings_initializer='he_normal',
            mask_zero=True
        )

        if init_biases_to_avgs:
            self.user_bias = Embedding(num_users + 1, 1, weights=[self.user_biases_init_to_avgs])
            self.movie_bias = Embedding(num_movies + 1, 1, weights=[self.movie_biases_init_to_avgs])
        else:
            self.user_bias = Embedding(num_users + 1, 1)
            self.movie_bias = Embedding(num_movies + 1, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = tf.squeeze(self.user_bias(inputs[:, 0]))

        movie_vector = self.movie_embedding(inputs[:, 1])
        movie_bias = tf.squeeze(self.movie_bias(inputs[:, 1]))

        global_bias = self.global_bias
        dot_user_movie = tf.tensordot(user_vector, movie_vector, 2)

        # Add all the components (including biases)
        return dot_user_movie + user_bias + movie_bias + global_bias


tuning_results_tracker = {}    # dictionary to track the results and ultimately return the best hyperparameters
lr_const_init = 1e-4           # lr will be reduced during the training process automatically when getting to a plateau


def build_model(init_biases_to_avgs: bool, embedding_dim: int, optimizer: optimizers,
                batch_size: int, is_hyperparameter_tuning: bool) -> Sequential:


    mf_model = MatrixFactorizationNet(init_biases_to_avgs=init_biases_to_avgs, n_users=num_users, n_movies=num_movies,
                                      embedding_dim=embedding_dim, global_bias=train_global_avg)

    mf_model.compile(loss='mean_squared_error', optimizer=optimizer(lr=lr_const_init))

    # CALLBACKS:
    # if the val_loss score does not improve within X epochs - stop training
    es = EarlyStopping(monitor='val_loss', verbose=1, patience=5)

    # if the val_loss does not improve in X epochs - divide the learning rate by 2. Stop dividing if reached 1e-8.
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-8, verbose=1)

    history = mf_model.fit(
        x=X_train, y=y_train,
        steps_per_epoch=X_train.shape[0] // batch_size,
        validation_steps=X_val.shape[0] // batch_size,
        batch_size=batch_size, callbacks=[es, reduce_lr],
        epochs=EPOCHS, verbose=1, validation_data=(X_val, y_val),
    )

    final_loss_score = history.history['val_loss'][-1]

    if is_hyperparameter_tuning:
        optimizer_name = str(optimizer).split('.')[-1][:-2]
        curr_hyperparameters = f'init_biases_to_avgs: {init_biases_to_avgs}, embedding_dim: {embedding_dim}, ' \
                               f'optimizer: {optimizer_name}, batch_size: {batch_size}'

        print(f'curr_try hyperparameters: {curr_hyperparameters}')
        print(f'MODEL LOSS: {final_loss_score}\n')
        tuning_results_tracker[curr_hyperparameters] = final_loss_score

    else:
        print(f'FINAL MODEL LOSS TO BE SAVED: {final_loss_score}')

        # save model
        mf_model.save('final_mf_model')

    _ = collect()
    return history


# set to False to train a regular network
is_hyperparameter_tuning = False


def main():

    if is_hyperparameter_tuning:
        n_tries = 10

        ### params grid ###
        is_init_biases_to_avgs = [True, False]
        embedding_dims = [16, 32, 48, 64, 96, 128]
        # LazyAdam == SparseAdam
        # Adamax -> version of Adam sometimes superior to adam, especially in models with embeddings
        optimizers = [LazyAdam, Adam, Adamax]
        batch_sizes = [16, 32, 64]

        for curr_try in range(n_tries):
            print(f'currunt try: {curr_try + 1}/{n_tries}')

            # randomly choose hyperparameters and train a model using them
            init_biases_to_avgs = random.choice(is_init_biases_to_avgs)
            emb_dim = random.choice(embedding_dims)
            optimizer = random.choice(optimizers)
            batch_size = random.choice(batch_sizes)

            build_model(init_biases_to_avgs=init_biases_to_avgs, embedding_dim=emb_dim, optimizer=optimizer,
                        batch_size=batch_size, is_hyperparameter_tuning=is_hyperparameter_tuning)

        best_hyperparameters = min(zip(tuning_results_tracker.values(), tuning_results_tracker.keys()))[1]
        print(f'THE BEST HYPERPARAMETERS ARE: {best_hyperparameters}')

    else:
        # Hyperparameters are tuned, let's train the final model on the train set and save it's epoch
        # with the best loss on the validation set
        best_init_biases_to_avgs = False
        best_emb_dim = 64
        best_optimizer = Adam
        best_batch_size = 64

        history = build_model(init_biases_to_avgs=best_init_biases_to_avgs, embedding_dim=best_emb_dim,
                              optimizer=best_optimizer, batch_size=best_batch_size,
                              is_hyperparameter_tuning=is_hyperparameter_tuning)


        # plot results
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.grid(linestyle='--', linewidth=0.5)
        plt.title("Matrix Factorization Model Loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["Train", "Val"], loc="upper left")
        plt.show()


if __name__ == "__main__":
    main()

