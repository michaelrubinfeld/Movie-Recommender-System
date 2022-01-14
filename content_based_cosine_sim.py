import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
import Levenshtein as lev
from collections import Counter
from typing import List, Any, Tuple


def cosine_sim_recommender(movies_df: pd.DataFrame, movies_list: List[str], how_many: int = 10) -> None:
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

        # deal with year without the title
        if year.isnumeric():
            return int(year)
        else:
            return np.nan

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

    '''
    We will use tf-idf in order to give appropriate weights to the movie genres, using the weights to calculate
    similarity between movies.
    
    We would like to take into account every combination of genres in all movies so we will calculate the tf-idf of 
    every single genre combination in each movie present in the dataframe.
    
    To conclude - we will calculate the weights of the superset of each movie's genres and then and keep all unique 
    values all over the dataframe.
    '''

    max_genre_ngram_range = 3     # best ngram found in training the content based model

    tf_idf_vector = TfidfVectorizer(analyzer=lambda genres: (combination for combination_range_size in
                                                             range(1, max_genre_ngram_range+1) for combination in
                                                             combinations(genres.split('|'),
                                                                          r=combination_range_size)),
                                    stop_words='english')

    tfidf_genre_combinations_matrix = tf_idf_vector.fit_transform(movies_df['genres'].values.astype(str))

    # now, we will calculate the cosine similarity between movies
    sim_matrix = cosine_similarity(tfidf_genre_combinations_matrix)

    '''
    On to the content based recommender system itself.
    First, we will create a function (using some helper functions) that applies Levenshtein Distance on two strings.
    This will help us get to the movie we want even if we misspell its name!
    '''

    # a function to convert index to title
    def get_title_from_index(index: int) -> str:
        return movies_df[movies_df.index == index]['title'].values[0]

    def get_leven_scores(movie_titles_list: List[str], title_to_compute_distance_from: str) -> List[tuple]:
        leven_scores = []
        for index, value in enumerate(movie_titles_list):
            leven_score = lev.ratio(title_to_compute_distance_from, value)
            leven_tuple = (index, leven_score)
            leven_scores.append(leven_tuple)
        return leven_scores

    # a function to return the most similar title to the words a user type and the distance score
    # it has from the raw input
    def find_closest_title(title: str) -> Tuple[str, int]:
        # list of tuples where each is (index, Levenshtein Distance) for all movies in the dataframe
        leven_scores = get_leven_scores(movies_df['title'].tolist(), title)

        # sort the list created by the Distances in reverse order
        sorted_leven_scores = sorted(leven_scores, key=lambda x: x[1], reverse=True)

        # get closest movie title to closest to the input title and its distance score and return them
        closest_title = get_title_from_index(sorted_leven_scores[0][0])
        distance_score = sorted_leven_scores[0][1]
        return closest_title, distance_score

    '''
    The function find_closest_title() will return the most similar title in the data to the words a user types.
    Without this, the recommender only works when a user enters the exact title which the data has.
    '''

    # helper functions for the recommender
    def get_title_year_from_index(index: int) -> str:
        return movies_df[movies_df.index == index]['title_year'].values[0]

    def get_index_from_title(title: str) -> int:
        return movies_df[movies_df.title == title].index.values[0]

    '''
    The content based recommender gets as input a movie and how many recommendations to present. The function only 
    prints recommendations based on similarity to the movie input
    '''

    def content_based_recommender(movie_user_likes: str, how_many: int) -> List[str]:
        closest_title, distance_ratio = find_closest_title(movie_user_likes)
        recommendations = []
        # When a user makes misspellings (Levenshtein Distance ratio is 1.0 when two strings are the same)
        if distance_ratio < 1.0:
            # prints the movie meant by the user
            print('Did you mean ' + '\033[1m' + str(closest_title) + '\033[0m' + '?', '\n')

        movie_index = get_index_from_title(closest_title)
        movie_list = list(enumerate(sim_matrix[int(movie_index)]))

        # get similar movies while removing the input movie itself
        similar_movies = list(
            filter(lambda x: x[0] != int(movie_index), sorted(movie_list, key=lambda x: x[1], reverse=True)))

        for i, s in similar_movies[:how_many]:
            recommendations.append(get_title_year_from_index(i))

        return recommendations


    all_recommendations = []
    for movie in movies_list:
        curr_recs = content_based_recommender(movie, how_many)
        all_recommendations.append(curr_recs)

    # upack list of lists to one long list
    all_recommendations = [item for sublist in all_recommendations for item in sublist]

    # take all movie recommendations and count their frequency, later sort by frequency in reverse
    recommendations_sorted = sorted(dict(Counter(all_recommendations)).items(), key=lambda item: item[1])[::-1]

    print('Showing recommendations for:')
    print("====" * 9)
    for movie in movies_list:
        closest_title, distance_ratio = find_closest_title(movie)
        if distance_ratio < 1.0:
            print(f'"{movie}" corrected to: {closest_title}')
        else:
            print(f'{movie}')
    print("----" * 8)
    print(f"Top {how_many} movie recommendations")
    print("----" * 8)
    for movie, freq in recommendations_sorted[:how_many]:
        movie = str(movie).rsplit(' ', 1)[0]
        movie_year = movies_df.loc[movies_df['title'] == movie, 'year'].values
        print(f'{movie} ({movie_year})')
    print("====" * 9)


def main():
    # Running Example
    movies_dataframe = pd.read_csv('movies.csv')
    movies = ['North by Northwest', 'Casblanca', 'Rebeca', 'Star Wars The empire Strikes Back', '12 Angry Men']
    cosine_sim_recommender(movies_dataframe, movies)


if __name__ == "__main__":
    main()

