import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



dataset_movies = pd.read_csv("movies.csv")
dataset_ratings = pd.read_csv("ratings.csv")

dataset_ratings = dataset_ratings.drop(['timestamp'] , axis = 1)
merged_df = pd.merge(dataset_ratings, dataset_movies , on = 'movieId')

combine_movie_rating = merged_df.dropna(axis = 0, subset = ['title'])

movie_ratingCount = (combine_movie_rating.groupby(by = ['title'])['rating'].
                     count().
                     reset_index().
                     rename(columns = {'rating' : 'totalRatingCount'})
                     [['title' , 'totalRatingCount']]
                     )

updated_df = combine_movie_rating.merge(movie_ratingCount , left_on = 'title' , right_on = 'title' , how = 'left')


pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(movie_ratingCount['totalRatingCount'].describe())

popularity_threshold = 40
rating_popular_movie = updated_df.query('totalRatingCount >= @popularity_threshold')



#using pandas library
dataset_pivoted = rating_popular_movie.pivot_table(index = ['title' ,'movieId'], columns = 'userId' , values = 'rating').fillna(0)

#This is easier to represent the data. The 2nd user has given ratings for the movies with ID 1,3,5
#but not for movies with ID 2,4 etc. We have made a matrix now where the location of a value is
#(Title, userID) and the value at that location is the rating provided by the userID for the movieID



from scipy.sparse import csr_matrix
#There will be a lot of 0's in our matrix so to consider only non-zero values we remove sparsity

csr_data = csr_matrix(dataset_pivoted.values)
#dataset_pivoted.reset_index(inplace = True)


from sklearn.neighbors import NearestNeighbors
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(csr_data) 

df2 = (rating_popular_movie.groupby(by = ['title'])['movieId'].
       mean().
       reset_index()
      )


n_movies = int(input("How many recommendations do you want?"))
movie_name = input("Enter Movie Name:")
print("\n\n")



movie_list = rating_popular_movie[rating_popular_movie['title'].str.contains(movie_name)] 
print(movie_list)

#if len(movie_list):       
movie_idx=int(movie_list.iloc[0]['movieId'])
print(movie_idx)





      
index1 = 0
for i in range(0,len(df2)):
    if(df2.iloc[i, 1]!= movie_idx):
        index1 +=1
    else:
        break





distances, indices = knn.kneighbors(dataset_pivoted.iloc[index1, :].values.reshape(1,-1), n_neighbors =n_movies + 1)

print(indices.flatten())
for i in range(0,len(distances.flatten())):
    if i == 0:
        print('Recommendations of {0}:\n'.format(dataset_pivoted.index[index1]))
    if i != 0:
        print('{0}:{1} with a distance of {2}:'.format(i,dataset_pivoted.index[indices.flatten()[i]],distances.flatten()[i]))















"""def get_movies(movie_name , n):
    n_movies = n
    movie_list = movies[movies['title'].str.contains(movie_name)]
    if(len(movie_list)):
        movie_idx= movie_list.iloc[0]['movieId']
        movie_idx = dataset_pivoted[dataset_pivoted['movieId'] == movie_idx].index[0]
        distance, indices = knn.kneighbors(dataset_pivoted.iloc[movie_idx, :].values.reshape(1,-1) , n_neighbors = n_movies+1)
        for i in range(0,len(distances.flatten())):
            if i == 0 :
                print('Recommendations for ' + movie_name + '\n')
                else:
                print(dataset_pivoted.index[indices.flatten()[i]])
    else:
        print('No film found')"""
  


