import pandas as pd
import numpy as np
import ast
import nltk
import json

import matplotlib.pyplot as plt
# import seaborn as sns

from nltk.stem.porter import PorterStemmer 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# from wordcloud import WordCloud, STOPWORDS
# import nltk
# from nltk.corpus import stopwordsplt.subplots(figsize=(12,12))
# stop_words = set(stopwords.words('english'))
# stop_words.update(',',';','!','?','.','(',')')
from scipy import spatial
import operator

def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i["name"])
    return L

# def convert3(obj):
#     L = []
#     counter = 0
#     for i in ast.literal_eval(obj):
#         if counter != 3:
#             L.append(i["name"])
#             counter += 1
#         else:
#             break     
#     return L

# def fetch_director(obj):
#     L = []
#     for i in ast.literal_eval(obj):
#         if i["job"] == "Director":
#             L.append(i["name"])
#             break
            
#     return L

# def stem(text):
#     y = []
#     for i in text.split():
#         y.append(ps.stem(i))
#     return " ".join(y)

def extract_names(json_string):
    names_list = []
    for item in ast.literal_eval(json_string):
        names_list.append(item["name"])
    return names_list

# def extract_names(input_data):
#     # Check if the input is a string and convert it to a list if necessary
#     if isinstance(input_data, str):
#         input_data = ast.literal_eval(input_data)
#     # Extract names from the list
#     names_list = [item["name"] for item in input_data]
#     return names_list

def extract_top_three_names(json_string):
    top_names = []
    count = 0
    for item in ast.literal_eval(json_string):
        if count < 3:
            top_names.append(item["name"])
            count += 1
        else:
            break
    return top_names

def get_director_name(json_string):
    director_list = []
    for person in ast.literal_eval(json_string):
        if person["job"] == "Director":
            director_list.append(person["name"])
            break
    return director_list

def apply_stemming(text):
    stemmed_words = []
    for word in text.split():
        stemmed_words.append(ps.stem(word))
    return " ".join(stemmed_words)


def recommend(movie):
    movie_index = new_df[new_df["title"] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
        # print(new_df.iloc[i[0]])

    return movies_list


def attribute_match(query_movie_id, recommended_movie_id, attribute):
    query_attrs = set(new_df[new_df['movie_id'] == query_movie_id][attribute].iloc[0].split())
    rec_attrs = set(new_df[new_df['movie_id'] == recommended_movie_id][attribute].iloc[0].split())
    intersection = query_attrs.intersection(rec_attrs)
    return len(intersection) / len(query_attrs)

def evaluate_recommendations(query_movie_id, recommended_ids):
    scores = []
    for rec_id in recommended_ids:
        score = attribute_match(query_movie_id, rec_id, 'tags')  # Using 'tags' which include genres, keywords, etc.
        scores.append(score)
    print(scores)
    return sum(scores) / len(scores)  # This is the average precision

    

def binary(specific_list, entire_list):
    binaryList = []
    
    for i in entire_list:
        if i in specific_list:
            binaryList.append(1)
        else:
            binaryList.append(0)
    
    return binaryList

def Similarity(movieId1, movieId2):
    # print()
    # print(movieId1)
    # print(movieId2)
    # print()
    a = movies[movies['id'] == movieId1].iloc[0]
    b = movies[movies['id'] == movieId2].iloc[0] 

    # print(a)
    
    genresA = a['genres_encoded']
    genresB = b['genres_encoded']
    
    genreDistance = spatial.distance.cosine(genresA, genresB)
    
    scoreA = a['actors_encoded']
    scoreB = b['actors_encoded']
    scoreDistance = spatial.distance.cosine(scoreA, scoreB)
    
    directA = a['directors_encoded']
    directB = b['directors_encoded']
    directDistance = spatial.distance.cosine(directA, directB)

    return (genreDistance + directDistance + scoreDistance)/3.0


# def predict_score():
#     name = input('Enter a movie title: ')
#     new_movie = movies[movies['original_title'].str.contains(name)].iloc[0].to_frame().T
#     print('Selected Movie: ',new_movie.original_title.values[0])

def getNeighbors(baseMovie, K):
    distances = []

    for index, movie in movies.iterrows():
        if movie['id'] != baseMovie['id'].values[0]:
            dist = Similarity(baseMovie['id'].values[0], movie['id'])
            distances.append((movie['id'], dist))

    distances.sort(key=operator.itemgetter(1))
    neighbors = []

    for x in range(K):
        neighbors.append(distances[x])
    return neighbors


def convert_list_items(data):
    return data.apply(lambda items: [item.replace(" ", "") for item in items])


if __name__ == "__main__":

    # ----------- reading data ---------------
    movies = pd.read_csv("../tbdm_5000/tmdb_5000_movies.csv")
    credits = pd.read_csv("../tbdm_5000/tmdb_5000_credits.csv")

    ## ----------- cosine similarity ---------------

    print("cosine similarity start")
    
    movies = movies.merge(credits,on = "title")
    movies = movies[["movie_id","title","overview","genres","keywords","cast","crew"]]

    movies = movies.dropna()

    movies["genres"] = movies["genres"].apply(extract_names)
    movies["keywords"] = movies["keywords"].apply(extract_names)
    movies["cast"] = movies["cast"].apply(extract_top_three_names)
    movies["crew"] = movies["crew"].apply(get_director_name)

    movies["overview"] = movies["overview"].apply(lambda x:x.split())
    movies["genres"] = movies["genres"].apply(lambda x:[i.replace(" ","")for i in x])
    movies["keywords"] = movies["keywords"].apply(lambda x:[i.replace(" ","")for i in x])
    movies["cast"] = movies["cast"].apply(lambda x:[i.replace(" ","")for i in x])
    movies["crew"] = movies["crew"].apply(lambda x:[i.replace(" ","")for i in x])

    movies["tags"] = movies["overview"] + movies["genres"] + movies["keywords"] + movies["cast"] + movies["crew"]

    new_df = movies[["movie_id","title","tags"]]
    new_df["tags"] = new_df["tags"].apply(lambda x:" ".join(x))
    new_df["tags"] = new_df["tags"].apply(lambda x:x.lower())

    ps = PorterStemmer()
    # new_df["tags"]  = new_df["tags"].apply(stem)

    # cv = CountVectorizer(max_features=5000,stop_words="english")

    # vectors = cv.fit_transform(new_df["tags"]).toarray()

    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    vectors = tfidf.fit_transform(new_df["tags"]).toarray()

    similarity = cosine_similarity(vectors)

    movies_list = recommend("Batman Begins")
    recommended_movie_ids = []

    for i in movies_list:
        recommended_movie_ids.append(new_df.iloc[i[0]].movie_id)

    query_movie_id = new_df[new_df['title'] == 'Batman Begins']['movie_id'].iloc[0]
    recommended_ids = [id for id in recommended_movie_ids]  # Assume you have a list of IDs from your recommend function
    print(evaluate_recommendations(query_movie_id, recommended_ids))

    print("cosine similarity done")

    # ----------- knn ---------------
    
    print("knn start")

    movies_knn = pd.read_csv("../tbdm_5000/tmdb_5000_movies.csv")
    credits_knn = pd.read_csv("../tbdm_5000/tmdb_5000_credits.csv")
    
    # Converting string representations into actual lists
    movies_knn["genre_data"] = movies_knn["genres"].apply(extract_names)
    movies_knn["keyword_data"] = movies_knn["keywords"].apply(extract_names)
    credits_knn["actor_list"] = credits_knn["cast"].apply(extract_top_three_names)
    credits_knn["director_data"] = credits_knn["crew"].apply(get_director_name)

    # Merging datasets
    combined_data = movies_knn.merge(credits_knn, left_on='id', right_on='movie_id', how='left')
    selected_columns = ['id', 'original_title', 'genre_data', 'actor_list', 'vote_average', 'director_data', 'keyword_data']
    movies = combined_data[selected_columns]

    # Removing spaces from list items
    movies["genre_data"] = convert_list_items(movies["genre_data"])
    movies["keyword_data"] = convert_list_items(movies["keyword_data"])
    movies["actor_list"] = convert_list_items(movies["actor_list"])
    movies["director_data"] = convert_list_items(movies["director_data"])

    # Collecting unique genres
    unique_genres = []
    for _, row in movies.iterrows():
        for genre in row["genre_data"]:
            if genre not in unique_genres:
                unique_genres.append(genre)

    movies['genres_encoded'] = movies['genre_data'].apply(lambda x: binary(x, unique_genres))

    # Collecting unique actors
    unique_actors = []
    for _, row in movies.iterrows():
        for actor in row["actor_list"]:
            if actor not in unique_actors:
                unique_actors.append(actor)

    movies['actors_encoded'] = movies['actor_list'].apply(lambda x: binary(x, unique_actors))

    # Collecting unique directors
    unique_directors = []
    for _, row in movies.iterrows():
        for director in row["director_data"]:
            if director not in unique_directors:
                unique_directors.append(director)

    movies['directors_encoded'] = movies['director_data'].apply(lambda x: binary(x, unique_directors))

    num_neighbors = 10
    average_rating = 0

    # Sampling a base movie
    sampled_movie = movies.sample(n=1)

    # Finding nearest neighbors
    neighborhood = getNeighbors(sampled_movie, num_neighbors)

    # Initialize a list to store results
    detailed_neighbors = []

    # Loop through each neighbor tuple
    for movie_id, rating_avg in neighborhood:
        # Retrieve the original_title for the corresponding movieId
        title = movies[movies['id'] == movie_id]['original_title'].values[0]
        genre = movies[movies['id'] == movie_id]['genre_data']
        # Append the details to the list
        detailed_neighbors.append((movie_id, title, rating_avg, genre))

    # Print results
    print("\n", sampled_movie['original_title'], sampled_movie['genre_data'])

    print("\n recommendations")
    for neighbor in detailed_neighbors:
        print(f"Movie ID: {neighbor[0]}, Title: {neighbor[1]}, Rating Avg: {neighbor[2]}, Genres: {neighbor[3]}")

    
    print("knn done")

