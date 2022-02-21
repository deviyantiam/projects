# The project is to find similar movies that user might like by designing recommendation systems. We are going to create a item-based recommendation system and a user-based recommendation system.

# # Preprocessing
# import all the frameworks that we need and all the data we want to process

import pandas as pd
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
class Model:
    def __init__(self):
        # self.input_id=None
        pass
    def bygenre(input_movies):
        df_movie= pd.read_csv('movies.csv')
        df_rating = pd.read_csv('ratings.csv')
# print(df_movie.head())
#print(df_movie.shape)
#print(df.movie.isna().sum())
# firstly, we need to :
# - extract the years in the 'title' column, create a new column named 'year' and move the years to the column
# - split values in the 'genres' column
#We specify the parantheses as year informations' format so we don't conflict with movies that have years in their titles
        df_movie['year'] = df_movie.title.str.extract('(\(\d\d\d\d\))',expand=False)
#Removing the parentheses
        df_movie['year'] = df_movie.year.str.extract('(\d\d\d\d)',expand=False)
#Removing the years from the 'title' column
        df_movie['title'] = df_movie.title.str.replace('(\(\d\d\d\d\))', '')
#Applying the strip function to get rid of any ending whitespace characters that may have appeared
        df_movie['title'] = df_movie['title'].apply(lambda x: x.strip())
#Splitting function on |
        df_movie['genres'] = df_movie.genres.str.split('|')
# print(df_movie.head())
# We also need to convert the list of genres to a vector where each column corresponds to one possible value of the feature (1 if movie was belong to some genre or 0 if not) by using one hot encoding.
        df_movgen = df_movie.copy()
        for index, row in df_movgen.iterrows():
            for gen in row['genres']:
                df_movgen.at[index, gen] = 1
#Filling in the NaN values with 0 to show that a movie doesn't have that column's genre
        df_movgen = df_movgen.fillna(0)
# print(df_movgen.head())
# Prepare the df_rating, delete column that we dont need.
        df_rating = df_rating.drop('timestamp', 1)
# print(df_rating.head())
# print(df_rating.shape)
# # Item based filter
# attempts to figure out the input's favorite genres from the movies and ratings given.
# Let's begin by creating an input user to recommend movies to:
# Notice: To add more movies, simply increase the amount of elements in the __user_input__. Feel free to add more in! Just be sure to write it in with capital letters and if a movie starts with a "The", like "The Breakfast Club" then write it in like this: 'Breakfast Club, The' .

# user_input = [
#             {'title':'Breakfast Club, The', 'rating':5},
#             {'title':'Toy Story', 'rating':3.5},
#             {'title':'Jumanji', 'rating':2},
#             {'title':"Pulp Fiction", 'rating':5},
#             {'title':'Akira', 'rating':4.5}
#          ] 
# input_movies = pd.DataFrame(user_input)

# With the input complete, let's extract the input movie's ID's from the movies dataframe and add them into it.
# We can achieve this by first filtering out the rows of df_movie that contain the input_movies's title and then merging this subset with the input_movies dataframe. We also drop unnecessary columns for the input to save memory space.
#Filtering out the movies by title
        input_id = df_movie[df_movie['title'].isin(input_movies.loc[:,'title'].tolist())]        
#Then merging it so we can get the movieId. It's implicitly merging it by title.
        input_movies = pd.merge(input_id, input_movies)
#Dropping information we won't use from the input dataframe
        input_movies = input_movies.drop('genres', 1).drop('year', 1)
#Final input dataframe
#If a movie you added in above isn't here, then it might not be in the original 
#dataframe or it might spelled differently, please check capitalisation.
# print(input_movies)
# We're going to start by learning the input's preferences, so let's get the subset of movies that the input has watched from the Dataframe containing genres defined with binary values.
# #Filtering out the movies from the input
        user_movies = df_movgen[df_movgen['movieId'].isin(input_movies['movieId'].tolist())]
# print(user_movies)
# We'll only need the actual genre table, so let's clean this up a bit by resetting the index and dropping the __movieId, title, genres__ and __year__ columns.
#Resetting the index to avoid future issues
        user_movies = user_movies.reset_index(drop=True)
#Dropping unnecessary issues due to save memory and to avoid issues
        userGenreTable = user_movies.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
# print(userGenreTable)
# Now we're ready to start learning the input's preferences!
# To do this, we're going to turn each genre into weights. We can do this by using the input's reviews and multiplying them into the input's genre table and then summing up the resulting table by column. This operation is actually a dot product between a matrix and a vector, so we can simply accomplish by calling Pandas's "dot" function.
# print(input_movies['rating'])
#Dot produt to get weights
        userProfile = userGenreTable.transpose().dot(input_movies['rating'])
#The user profile
# print(userProfile)
# Now, we have the weights for every of the user's preferences. This is known as the User Profile ( __userProfile__ ). Using this, we can recommend movies that satisfy the user's preferences.
# Let's start by extracting the genre table from the original dataframe:
#Now let's get the genres of every movie in our original dataframe
        genreTable = df_movgen.set_index(df_movgen['movieId'])
#And drop the unnecessary information
        genreTable = genreTable.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
# print(genreTable.head())
# print(genreTable.shape)
#Multiply the genres by the weights and then take the weighted average
        recommendationTable_df = ((genreTable*userProfile).sum(axis=1))/(userProfile.sum())
# print(recommendationTable_df.head())
# With the input's profile and the complete list of movies and their genres in hand, we're going to take the weighted average of every movie based on the input profile and recommend the top 5 movies that most satisfy it.
#Sort our recommendations in descending order
        recommendationTable_df = recommendationTable_df.sort_values(ascending=False)
#Just a peek at the values
# print(recommendationTable_df.head())
#The final recommendation table
        x=recommendationTable_df.head(5).keys()
## df_movie.query('movieId in @x')
## df_movie[df_movie['movieId'].isin(recommendationTable_df.head(5).keys())]
##### BY ORDER
        mov_byitem=df_movie.copy()
        mov_byitem=mov_byitem.set_index('movieId').loc[x].reset_index(inplace=False)
# print(mov_byitem)
        rate=df_rating[df_rating['movieId'].isin(mov_byitem['movieId'].tolist())].groupby('movieId')['rating'].sum()/df_rating[df_rating['movieId'].isin(mov_byitem['movieId'].tolist())].groupby('movieId')['rating'].count()
        rate_pd=pd.DataFrame(rate)
# print(rate_pd.head())
        mov_byitem=mov_byitem.merge(rate_pd, left_on='movieId', right_on='movieId')     
# print('MOVIES YOU MIGHT LIKE')
# print(mov_byitem)
        return mov_byitem


# # User based filter
# attempts to find users that have similar preferences and opinions as the input and then recommends items that they have liked to the input. We will be using a method based on the __Pearson Correlation Function__.
# The process for creating a User Based recommendation system is as follows:
# - Select a user with the movies the user has watched
# - Based on his rating to movies, find the top X neighbours 
# - Get the watched movie record of the user for each neighbour.
# - Calculate a similarity score using some formula
# - Recommend the items with the highest score
# Let's begin by creating an input user to recommend movies to:

# userInput = [
#             {'title':'Breakfast Club, The', 'rating':5},
#             {'title':'Toy Story', 'rating':3.5},
#             {'title':'Jumanji', 'rating':2},
#             {'title':"Pulp Fiction", 'rating':5},
#             {'title':'Akira', 'rating':4.5}
#          ] 
# inputMovies = pd.DataFrame(userInput)
class Model2:
    def __init__(self):
        # self.input_id=None
        pass
    def byusers(inputMovies):
        df_movie= pd.read_csv('movies.csv')
        df_rating = pd.read_csv('ratings.csv')
        df_movie['year'] = df_movie.title.str.extract('(\(\d\d\d\d\))',expand=False)
        df_movie['year'] = df_movie.year.str.extract('(\d\d\d\d)',expand=False)
        df_movie['title'] = df_movie.title.str.replace('(\(\d\d\d\d\))', '')
        df_movie['title'] = df_movie['title'].apply(lambda x: x.strip())
        df_movie['genres'] = df_movie.genres.str.split('|')
        df_rating = df_rating.drop('timestamp', 1)
        inputId = df_movie[df_movie['title'].isin(inputMovies['title'].tolist())]
#Then merging it so we can get the movieId. It's implicitly merging it by title.
        inputMovies = pd.merge(inputId, inputMovies)
#Dropping information we won't use from the input dataframe
        inputMovies = inputMovies.drop('year', 1)
        inputMovies = inputMovies.drop('genres', 1)
#Final input dataframe
#If a movie you added in above isn't here, then it might not be in the original 
#dataframe or it might spelled differently, please check capitalisation.
# print(inputMovies)
# Now with the movie ID's in our input, we can now get the subset of users that have watched and reviewed the movies in our input.
#Filtering out users that have watched movies that the input has watched and storing it
        userSubset = df_rating[df_rating['movieId'].isin(inputMovies['movieId'].tolist())]
# print(userSubset.head())
# We now group up the rows by user ID.
        userSubsetGroup = userSubset.groupby(['userId'])
# Lets look at one of the users, e.g. the one with userID=1607
# print(len(userSubsetGroup))
# check some row
# print(userSubsetGroup.get_group(1607))
# Let's also sort these groups so the users that share the most movies in common with the input have higher priority. This provides a richer recommendation since we won't go through every single user.
        userSubsetGroup = sorted(userSubsetGroup,  key=lambda x: len(x[1]), reverse=True)
# Now lets look at the first 5 users
# print(userSubsetGroup[0:5])
# #### Similarity of users to input user
# Next, we are going to compare some users to our specified user and find the one that is most similar.  
# we're going to find out how similar each user is to the input through the __Pearson Correlation Coefficient__. It is used to measure the strength of a linear association between two variables.
# ![alt text](https://wikimedia.org/api/rest_v1/media/math/render/svg/bd1ccc2979b0fd1c1aec96e386f686ae874f9ec0 "Pearson Correlation")
# In our case, a 1 means that the two users have similar tastes while a -1 means the opposite.
# We will select a subset of users to iterate through. This limit is imposed because we don't want to waste too much time going through every single user.
        userSubsetGroup = userSubsetGroup[0:500]
# Now, we calculate the Pearson Correlation between input user and subset group, and store it in a dictionary
#Store the Pearson Correlation in a dictionary, where the key is the user Id and the value is the coefficient
        pearsonCorrelationDict = {}
#For every user group in our subset
        for name, group in userSubsetGroup:
    #Let's start by sorting the input and current user group so the values aren't mixed up later on
            group = group.sort_values(by='movieId')
            inputMovies = inputMovies.sort_values(by='movieId')
    #Get the N for the formula
            nRatings = len(group)
    #Get the review scores for the movies that they both have in common
            temp_df = inputMovies[inputMovies['movieId'].isin(group['movieId'].tolist())]
    #And then store them in a temporary buffer variable in a list format to facilitate future calculations
            tempRatingList = temp_df['rating'].tolist()
    #Let's also put the current user group reviews in a list format
            tempGroupList = group['rating'].tolist()
    #Now let's calculate the pearson correlation between two users, so called, x and y
            Sxx = sum([i**2 for i in tempRatingList]) -(pow(sum(tempRatingList),2)/float(nRatings))
            Syy = sum([i**2 for i in tempGroupList]) - (pow(sum(tempGroupList),2)/float(nRatings))
            Sxy = sum( i*j for i, j in zip(tempRatingList, tempGroupList)) - (sum(tempRatingList)*sum(tempGroupList)/float(nRatings))
    #If the denominator is different than zero, then divide, else, 0 correlation.
            if Sxx != 0 and Syy != 0:
                pearsonCorrelationDict[name] = Sxy/sqrt(Sxx*Syy)
            else:
                pearsonCorrelationDict[name] = 0
        pearsonDF = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')
        pearsonDF.columns = ['similarityIndex']
        pearsonDF['userId'] = pearsonDF.index
        pearsonDF.index = range(len(pearsonDF))
# print(pearsonDF.head())
# Now let's get the top 50 users that are most similar to the input.
        topUsers=pearsonDF.sort_values(by='similarityIndex', ascending=False)[0:50]
# print(topUsers.head())
# Now, let's start recommending movies to the input user.
# We're going to do this by taking the weighted average of the ratings of the movies using the Pearson Correlation as the weight. But to do this, we first need to get the movies watched by the users in our __pearsonDF__ from the ratings dataframe and then store their correlation in a new column called _similarityIndex_. This is achieved below by merging of these two tables.
        topUsersRating=topUsers.merge(df_rating, left_on='userId', right_on='userId', how='inner')
# print(topUsersRating.head())
# Now all we need to do is simply multiply the movie rating by its weight (The similarity index), then sum up the new ratings and divide it by the sum of the weights.
# We can easily do this by simply multiplying two columns, then grouping up the dataframe by movieId and then dividing two columns:
# It shows the idea of all similar users to candidate movies for the input user:
#Multiplies the similarity by the user's ratings
        topUsersRating['weightedRating'] = topUsersRating['similarityIndex']*topUsersRating['rating']
# print(topUsersRating.head())
#Applies a sum to the topUsers after grouping it up by userId
        tempTopUsersRating = topUsersRating.groupby('movieId').sum()[['similarityIndex','weightedRating']]
        tempTopUsersRating.columns = ['sum_similarityIndex','sum_weightedRating']
# print(tempTopUsersRating.head())
#Creates an empty dataframe
        df_rec = pd.DataFrame()
#Now we take the weighted average
        df_rec['weighted average recommendation score'] = tempTopUsersRating['sum_weightedRating']/tempTopUsersRating['sum_similarityIndex']
# df_rec['movieId'] = tempTopUsersRating.index
# print(df_rec.head())
# Now let's sort it and see the top 20 movies that the algorithm recommended!
        df_rec = df_rec.sort_values(by='weighted average recommendation score', ascending=False)
# print(df_rec.head(10))
# df_movie.loc[df_movie['movieId'].isin(df_rec.head(10)['movieId'].tolist())]
## BY ORDER
# xy=df_rec.head(5).index.tolist()
# df_mvx=df_movie.copy()
# df_mvx=df_mvx[df_mvx['movieId'].isin(xy)]
        df_mvu=df_movie.copy()
        df_mvu=df_rec.merge(df_mvu,left_on='movieId',right_on='movieId',how='inner')
# print(df_mvu.head())
        rat=df_rating[df_rating['movieId'].isin(df_mvu['movieId'].tolist())].groupby('movieId')['rating'].sum()/df_rating[df_rating['movieId'].isin(df_mvu['movieId'].tolist())].groupby('movieId')['rating'].count()
        rate_us=pd.DataFrame(rat)
        df_mvu=df_mvu.merge(rate_us,left_on='movieId',right_on='movieId')
        df_mvu_new=df_mvu.sort_values(by=['weighted average recommendation score','rating'],ascending=[False,False]).reset_index(inplace=False)
        df_mvu_new=df_mvu_new.drop(['index'],axis=1)
        df_mvu_rec=df_mvu_new.loc[0:5,('movieId','title','genres','year','rating')]
# print('Other people who watched the movie(s) also watched and liked')
# print(df_mvu_rec.head())
        return df_mvu_rec
# if __name__ == '__main__':


# Conclusion:
# Item based technique is higliy personalised for the user, but it does not take into account what others think of the item. User based technique takes other user's ratings into consideration and adapts to the user's interests which might change over time, but it may takes a longer time to process the system