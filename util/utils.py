# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
##数据预处理
def panda_one_hot(column,need_df):
    classes = set(pd.unique(need_df[column]))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, need_df[column].values)),
                             dtype=np.int32)
    onehot_pd = pd.DataFrame(labels_onehot,columns = [i for i in classes_dict.keys()])
    return onehot_pd
def panda_many_hot(set_genres, need_df):
    classes = set(set_genres)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    genres_array = []
    for i in need_df.index:
        genres_array.append(np.array(list(map(classes_dict.get, need_df['genres'].values[i]))).sum(axis = 0))
    labels_onehot = np.array(genres_array)
    onehot_pd = pd.DataFrame(labels_onehot,columns = [i for i in classes_dict.keys()])
    return onehot_pd
def load_data():
    #gender,age,occupationid都是分类变量我们做one-hot处理
    #zip-code没用，删除
    users_columns = ['userid', 'gender', 'age', 'occupationid', 'zip-code']
    users = pd.read_csv('./data/users.dat', sep='::', header=None, names=users_columns, engine='python')
    
    gender = panda_one_hot('gender', users)
    age = panda_one_hot('age', users)
    occupationid = panda_one_hot('occupationid', users)
    users = pd.concat([users, gender, age, occupationid], axis = 1)
    users = users.drop(columns = ['gender', 'age', 'occupationid', 'zip-code'],axis = 1)
    
    #movies的genres做many-hot处理
    movies_columns = ['movieid', 'title', 'genres']
    movies = pd.read_csv('./data/movies.dat', sep='::', header=None, names=movies_columns, engine = 'python')
    movies['genres'] = movies['genres'].apply(lambda x: x.split("|"))
    set_genres = []
    for i in movies.index:
        set_genres += movies['genres'].iloc[i]
    genres = panda_many_hot(set_genres, movies)
    movies = pd.concat([movies, genres], axis = 1)
    #movies = movies.drop(columns = ['genres'], axis = 1) 不删除，可以embdding后使用
    
    ratings_columns = ['userid','movieid', 'rating', 'timestamp']
    ratings = pd.read_csv('./data/ratings.dat', sep='::', header=None, names=ratings_columns, engine = 'python')
    
    return users, movies, ratings

def load_data_embdding():
    # 用于做embedding
    # zip-code没用，删除
    users_columns = ['userid', 'gender', 'age', 'occupationid', 'zip-code']
    users = pd.read_csv('./data/users.dat', sep='::', header=None, names=users_columns, engine='python')

    # gender = panda_one_hot('gender', users)
    # age = panda_one_hot('age', users)
    # occupationid = panda_one_hot('occupationid', users)
    # users = pd.concat([users, gender, age, occupationid], axis = 1)
    users = users.drop(columns=['zip-code'], axis=1)

    movies_columns = ['movieid', 'title', 'genres']
    movies = pd.read_csv('./data/movies.dat', sep='::', header=None, names=movies_columns, engine='python')
    movies['genres'] = movies['genres'].apply(lambda x: x.split("|"))

    ratings_columns = ['userid', 'movieid', 'rating', 'timestamp']
    ratings = pd.read_csv('./data/ratings.dat', sep='::', header=None, names=ratings_columns, engine='python')

    return users, movies, ratings