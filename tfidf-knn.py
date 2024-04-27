import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


def load_data(movie_path, rating_path):
    data_movie = pd.read_csv(movie_path)
    data_rating = pd.read_csv(rating_path)
    movie = data_movie.loc[:, ["movieId", "title","genres"]]
    rating = data_rating.loc[:, ["userId", "movieId", "rating"]]
    return pd.merge(movie, rating)

# def create_pivot_table(data):
#     return data.pivot_table(index=["title"], columns=["userId"], values="rating").fillna(0)
def create_pivot_table(data):
    user_movie_table = data.pivot_table(index=["title"], columns=["userId"], values="rating").fillna(0)
    movie_genres = data.drop_duplicates(subset="title").set_index("title")["genres"]
    return user_movie_table, movie_genres


def train_model(user_movie_table):
    user_movie_table_matrix = csr_matrix(user_movie_table.values)
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(user_movie_table_matrix)
    return model_knn

# def get_recommendations(model_knn, user_movie_table, query_index):
#     distances, indices = model_knn.kneighbors(user_movie_table.iloc[query_index,:].values.reshape(1,-1), n_neighbors=6)
#     movie_recommendations = []
#     distance_scores = []
#     for i in range(1, len(distances.flatten())):
#         movie_recommendations.append(user_movie_table.index[indices.flatten()[i]])
#         distance_scores.append(distances.flatten()[i])
#     return pd.DataFrame({'movie': movie_recommendations, 'distance': distance_scores})

def get_recommendations(model_knn, user_movie_table, movie_genres, query_index):
    distances, indices = model_knn.kneighbors(user_movie_table.iloc[query_index,:].values.reshape(1,-1), n_neighbors=6)
    movie_recommendations = []
    distance_scores = []
    genres = []
    for i in range(1, len(distances.flatten())):
        title = user_movie_table.index[indices.flatten()[i]]
        movie_recommendations.append(title)
        distance_scores.append(distances.flatten()[i])
        genres.append(movie_genres.loc[title])
    return pd.DataFrame({'movie': movie_recommendations, 'distance': distance_scores, 'genre': genres})


def knn(movies_path, ratings_path):
    # Load and process data
    data = load_data(movies_path, ratings_path)
    data = data.iloc[:1000000, :]  # limit data for performance reasons
    user_movie_table, movie_genres = create_pivot_table(data)

    # Train model
    model_knn = train_model(user_movie_table)

    # Print all titles to check for the exact name
    print(user_movie_table.index.tolist())


    # Select random movie and get recommendations
    query_index = np.random.choice(user_movie_table.shape[0])
    # query_index = user_movie_table.index.get_loc("Ronin (1998)")

    recommendations = get_recommendations(model_knn, user_movie_table, movie_genres, query_index)

    # Display recommendations
    chosen_movie = user_movie_table.index[query_index]
    print('Recommendations for "{}":\n'.format(chosen_movie))
    for i in range(recommendations.shape[0]):
        print('{0}. title: {1}, distance: {2}, genre: {3}'.format(i, recommendations["movie"].iloc[i], recommendations["distance"].iloc[i], recommendations["genre"].iloc[i]))


# def get_average_similarity_two(movie_id):
#     idx = indices[movie_id]
#     sim_scores = cosine_sim[idx, :]
#     return sim_scores.mean()


# def get_average_similarity(movie_id):
#     if movie_id in movies['movieId'].values:
#         idx = indices[movies[movies['movieId'] == movie_id].index[0]]
#         sim_scores = cosine_sim[idx, :]
#         return sim_scores.mean()
#     else:
#         return 0

def enhanced_recommendations(movie_title, user_movies, cosine_sim, movie_indices, movie_data):
    
    if movie_title not in movie_indices:
        return "Movie not found in the dataset."
    
    idx = movie_indices[movie_title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    # Compute correlations with the target movie
    try:
        movie_similarities = user_movies.corrwith(user_movies[movie_title]).fillna(0)
    except KeyError:
        return "Movie not rated by enough users to form correlations."
    # print("\nMovie Similarities:", movie_similarities)

    # Adjust correlations with genre similarities
    for title in movie_similarities.index:
        if title in movie_indices:
            movie_data_idx = movie_indices[title]
            # Ensure genre_similarity is a scalar
            genre_similarity = cosine_sim[idx][movie_data_idx]
            # The error might be happening here if the assignment expects a scalar but gets an array
            # We need to ensure that we are working with a scalar value
            if isinstance(genre_similarity, np.ndarray) and genre_similarity.size == 1:
                genre_similarity = float(genre_similarity)
            elif isinstance(genre_similarity, (int, float)):
                genre_similarity = genre_similarity
            else:
                continue  # Skip if we cannot resolve the scalar issue
            
            movie_similarities.at[title] *= (genre_similarity + 1) / 2  # Corrected scaling and combining

    # sim_scores = movie_similarities.sort_values(ascending=False)
    sim_scores = movie_similarities.sort_values(ascending=False).head(10)
    recommended_indices = sim_scores.index.tolist()
    recommended_movies_with_genres = movie_data.loc[movie_data['title'].isin(recommended_indices), ['title', 'genres']]
    return recommended_movies_with_genres


    return sim_scores.head(10)


if __name__ == '__main__':

    movies_path = "../20mil_dataset/movie.csv"
    ratings_path = "../20mil_dataset/rating.csv"

    print("\n-----------knn model-------------\n")
    knn(movies_path, ratings_path)
    print("\n-----------knn done-------------\n")

    
    #------------ tfidf user-based -------------
    
    print("\n-----------tfidf model-------------\n")

    print("Loading data...\n")

    # Load movie and rating data
    movie_data = pd.read_csv("../20mil_dataset/movie.csv")
    rating_data = pd.read_csv("../20mil_dataset/rating.csv")

    print("Merging data...\n")
    # Combine movie details with user ratings
    merged_data = movie_data.merge(rating_data, on="movieId")

    print(merged_data.head(5))

    print("Filtering data...\n")
    # Determine popular movies based on a threshold of ratings received
    popularity_threshold = 1000
    movie_popularity = pd.DataFrame(merged_data['title'].value_counts())
    popular_movies = movie_popularity[movie_popularity['title'] > popularity_threshold].index
    filtered_movies = merged_data[merged_data['title'].isin(popular_movies)]

    print("tfidf user-based model result")

    # Create a matrix of user ratings for movies
    user_ratings_matrix = filtered_movies.pivot_table(index="userId", columns="title", values="rating")

    print("Selecting a user...\n")
    # Randomly select a user for recommendation
    # selected_user = int(pd.Series(user_ratings_matrix.index).sample(1, random_state=45).values)
    selected_user = int(pd.Series(user_ratings_matrix.index).sample(1).values)
    print(selected_user)
    user_ratings = user_ratings_matrix.loc[selected_user]

    # Identify movies rated by the selected user
    watched_movie_titles = user_ratings.dropna().index.tolist()
    watched_movie_matrix = user_ratings_matrix[watched_movie_titles]

    print("Analyzing similar users...\n")
    # Find users with similar viewing history
    similarity_threshold = 20
    similar_users_count = watched_movie_matrix.T.notnull().sum()
    similar_users = similar_users_count[similar_users_count > similarity_threshold].index

    # Correlation matrix calculation
    correlation_matrix = pd.concat([
        watched_movie_matrix.loc[similar_users],
        watched_movie_matrix.loc[[selected_user]]
    ]).T.corr().unstack().sort_values().drop_duplicates()

    # Format the correlation data
    formatted_corr = pd.DataFrame(correlation_matrix, columns=['correlation'])
    formatted_corr.index.names = ['base_user', 'compared_user']
    formatted_corr = formatted_corr.reset_index()

    print("Identifying top correlated users...\n")
    # Filter for highly correlated users
    correlation_cutoff = 0.65
    # top_correlated_users = formatted_corr[
    #     (formatted_corr['base_user'] == selected_user) &
    #     (formatted_corr['correlation'] > correlation_cutoff)
    # ].sort_values(by='correlation', ascending=False)['compared_user']

    # # Gather top user ratings
    # top_user_ratings = top_correlated_users.merge(rating_data, on="userId")

    # Filtering for highly correlated users and keeping it as DataFrame
    # top_correlated_users = formatted_corr[
    #     (formatted_corr['base_user'] == selected_user) &
    #     (formatted_corr['correlation'] > correlation_cutoff)
    # ].sort_values(by='correlation', ascending=False)[['compared_user']].rename(columns={'compared_user': 'userId'})

    # # Now merge should work since top_correlated_users is a DataFrame with a column named 'userId'
    # top_user_ratings = top_correlated_users.merge(rating_data, on="userId")

    # weighted_ratings = top_user_ratings[top_user_ratings['userId'] != selected_user]
    # weighted_ratings['weighted_score'] = weighted_ratings['correlation'] * weighted_ratings['rating']

    # Ensure that the DataFrame includes both 'compared_user' and 'correlation' columns
    top_correlated_users = formatted_corr[
        (formatted_corr['base_user'] == selected_user) &
        (formatted_corr['correlation'] > correlation_cutoff)
    ].sort_values(by='correlation', ascending=False)[['compared_user', 'correlation']]

    # Rename columns appropriately for subsequent merge operation
    top_correlated_users = top_correlated_users.rename(columns={'compared_user': 'userId'})

    # Merge with rating data, ensuring the 'correlation' column is included
    top_user_ratings = top_correlated_users.merge(rating_data, on="userId")

    # Ensure we do not include the selected user's own ratings in the weighted calculation
    weighted_ratings = top_user_ratings[top_user_ratings['userId'] != selected_user]

    # Now calculate the weighted ratings
    weighted_ratings['weighted_score'] = weighted_ratings['correlation'] * weighted_ratings['rating']


    print("Calculating recommendations...\n")
    # Aggregate weighted ratings
    recommendation_scores = weighted_ratings.groupby('movieId').agg({"weighted_score": "mean"})
    recommendation_scores = recommendation_scores.reset_index()

    # TF-IDF and cosine similarity for content-based filtering
    tfidf_vectorizer = TfidfVectorizer(token_pattern='(?u)\\b\\w+\\b')
    genre_matrix = tfidf_vectorizer.fit_transform(movie_data['genres'].str.replace('|', ' '))
    content_similarity = cosine_similarity(genre_matrix)

    # Content similarity scores
    movie_indices = pd.Series(movie_data.index, index=movie_data['movieId']).drop_duplicates()
    recommendation_scores['content_score'] = recommendation_scores['movieId'].apply(
        lambda x: content_similarity[movie_indices[x], :].mean()
    )

    # Final score calculation
    recommendation_scores['final_recommendation_score'] = (
        0.5 * recommendation_scores['weighted_score'] + 
        0.5 * recommendation_scores['content_score']
    )

    # Get top 5 recommendations
    top_recommendations = recommendation_scores.nlargest(5, 'final_recommendation_score')

    print("Finalizing recommendations...\n")
    # Merge recommendations with movie titles
    recommended_movies = top_recommendations.merge(movie_data[['movieId', 'title', 'genres']], on='movieId')
    print(recommended_movies)

    print("\n ------------tfidf user-based done----------------\n")


    #------------ tfidf item-based -------------
    print("\n---------------tfidf item-based model result---------------\n")


    print()
    print("movie data shape = ", movie_data.shape)

    user_movies = filtered_movies.pivot_table(index="userId", columns="title", values="rating")

    tfidf = TfidfVectorizer(stop_words='english')
    movie_data['genres'] = movie_data['genres'].fillna('')
    tfidf_matrix = tfidf.fit_transform(movie_data['genres'])

    # Compute cosine similarity matrix for the genres
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    movie_indices = pd.Series(movie_data.index, index=movie_data['title']).drop_duplicates()

    # print(movie_indices)

    # movie_name1 = "Matrix, The (1999)"
    # movie_name = user_movies[movie_name1]

    # final_rec = user_movies.corrwith(movie_name).sort_values(ascending=False).head(10)

    # movie_indices = pd.Series(movie_data.index, index=movie_data['title']).drop_duplicates()

    movie_name = "Ronin (1998)"

    # final_list = enhanced_recommendations(movie_name, user_movies, cosine_sim, movie_indices)

    # print(final_list)

    # movie_data.set_index('title', inplace=True)  # Set the index to 'title' if not already done
    final_list = enhanced_recommendations(movie_name, user_movies, cosine_sim, movie_indices, movie_data)
    print(final_list)






