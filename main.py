import numpy as np
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
movies = pd.read_csv('movies.csv')
credits = pd.read_csv('credits.csv')

def convertObjToString(obj):
    y = []
    for i in ast.literal_eval(obj):
        y.append(i['name'])
    return y

def getActorName(obj):
    y = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter <= 2:
            y.append(i['name'])
            counter += 1
    return y


def getDirectorName(obj):
    y = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            y.append(i['name'])
            break
    return y

def stem(text):
    y = []

    for i in text.split():
        y.append(ps.stem(i))   
    return " ".join(y)

def recommend (movie):
    movie_index = processed[processed['title']== movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)),reverse=True, key=lambda x:x[1])[1:6]
    for i in movie_list:
        print(processed.iloc[i[0]].title)
        # print(i[0])
    return

movies = movies.merge(credits, on='title')
movies= movies[['movie_id','title','overview','genres','keywords','cast','crew']]
movies.dropna(inplace=True)

movies['genres'] = movies['genres'].apply(convertObjToString)
movies['keywords'] = movies['keywords'].apply(convertObjToString)
movies['cast'] = movies['cast'].apply(getActorName)
movies['crew'] = movies['crew'].apply(getDirectorName)

movies['overview'] = movies['overview'].apply(lambda x:x.split())
movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ", "") for i in x])

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


processed = movies[['movie_id', 'title', 'tags']]
processed['tags'] = processed['tags'].apply(lambda x:" ".join(x))

processed['tags'] = processed['tags'].apply(lambda x: x.lower())

cv = CountVectorizer(max_features=8000,stop_words='english')
vectors = cv.fit_transform(processed['tags']).toarray()

processed['tags'] = processed['tags'].apply(stem)

similarity  = cosine_similarity(vectors)


recommend('The Avengers')

