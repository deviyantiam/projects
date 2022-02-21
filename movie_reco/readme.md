
The project is to find similar movies that user might like by designing recommendation systems. We are going to create a item-based recommendation system and a user-based recommendation system. Dataset can be downloaded at https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/moviedataset.zi

# Preprocessing
import all the frameworks that we need and all the data we want to process


```python
import pandas as pd
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
df_movie= pd.read_csv('data-rec/movies.csv')
df_rating = pd.read_csv('data-rec/ratings.csv')
df_movie.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children|Fantasy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy|Romance</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale (1995)</td>
      <td>Comedy|Drama|Romance</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II (1995)</td>
      <td>Comedy</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_movie.shape
```




    (34208, 3)




```python
df_movie.isna().sum()
```




    movieId    0
    title      0
    genres     0
    dtype: int64



firstly, we need to :
- extract the years in the 'title' column, create a new column named 'year' and move the years to the column
- split values in the 'genres' column


```python
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
df_movie.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story</td>
      <td>[Adventure, Animation, Children, Comedy, Fantasy]</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji</td>
      <td>[Adventure, Children, Fantasy]</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men</td>
      <td>[Comedy, Romance]</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale</td>
      <td>[Comedy, Drama, Romance]</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II</td>
      <td>[Comedy]</td>
      <td>1995</td>
    </tr>
  </tbody>
</table>
</div>



We also need to convert the list of genres to a vector where each column corresponds to one possible value of the feature (1 if movie was belong to some genre or 0 if not) by using one hot encoding.


```python
df_movgen = df_movie.copy()
for index, row in df_movgen.iterrows():
    for gen in row['genres']:
        df_movgen.at[index, gen] = 1
#Filling in the NaN values with 0 to show that a movie doesn't have that column's genre
df_movgen = df_movgen.fillna(0)
df_movgen.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>year</th>
      <th>Adventure</th>
      <th>Animation</th>
      <th>Children</th>
      <th>Comedy</th>
      <th>Fantasy</th>
      <th>Romance</th>
      <th>...</th>
      <th>Horror</th>
      <th>Mystery</th>
      <th>Sci-Fi</th>
      <th>IMAX</th>
      <th>Documentary</th>
      <th>War</th>
      <th>Musical</th>
      <th>Western</th>
      <th>Film-Noir</th>
      <th>(no genres listed)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story</td>
      <td>[Adventure, Animation, Children, Comedy, Fantasy]</td>
      <td>1995</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji</td>
      <td>[Adventure, Children, Fantasy]</td>
      <td>1995</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men</td>
      <td>[Comedy, Romance]</td>
      <td>1995</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale</td>
      <td>[Comedy, Drama, Romance]</td>
      <td>1995</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II</td>
      <td>[Comedy]</td>
      <td>1995</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>



Prepare the df_rating, delete column that we dont need.


```python
df_rating = df_rating.drop('timestamp', 1)
df_rating.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>169</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2471</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>48516</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>2571</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>109487</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_rating.shape
```




    (22884377, 3)



# Item based filter

attempts to figure out the input's favorite genres from the movies and ratings given.

Let's begin by creating an input user to recommend movies to:

Notice: To add more movies, simply increase the amount of elements in the __user_input__. Feel free to add more in! Just be sure to write it in with capital letters and if a movie starts with a "The", like "The Breakfast Club" then write it in like this: 'Breakfast Club, The' .


```python
user_input = [
            {'title':'Breakfast Club, The', 'rating':5},
            {'title':'Toy Story', 'rating':3.5},
            {'title':'Jumanji', 'rating':2},
            {'title':"Pulp Fiction", 'rating':5},
            {'title':'Akira', 'rating':4.5}
         ] 
input_movies = pd.DataFrame(user_input)
```

With the input complete, let's extract the input movie's ID's from the movies dataframe and add them into it.

We can achieve this by first filtering out the rows of df_movie that contain the input_movies's title and then merging this subset with the input_movies dataframe. We also drop unnecessary columns for the input to save memory space.


```python
#Filtering out the movies by title
input_id = df_movie[df_movie['title'].isin(input_movies['title'].tolist())]
#Then merging it so we can get the movieId. It's implicitly merging it by title.
input_movies = pd.merge(input_id, input_movies)
#Dropping information we won't use from the input dataframe
input_movies = input_movies.drop('genres', 1).drop('year', 1)
#Final input dataframe
#If a movie you added in above isn't here, then it might not be in the original 
#dataframe or it might spelled differently, please check capitalisation.
input_movies
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>296</td>
      <td>Pulp Fiction</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1274</td>
      <td>Akira</td>
      <td>4.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1968</td>
      <td>Breakfast Club, The</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>



We're going to start by learning the input's preferences, so let's get the subset of movies that the input has watched from the Dataframe containing genres defined with binary values.


```python
#Filtering out the movies from the input
user_movies = df_movgen[df_movgen['movieId'].isin(input_movies['movieId'].tolist())]
user_movies
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>year</th>
      <th>Adventure</th>
      <th>Animation</th>
      <th>Children</th>
      <th>Comedy</th>
      <th>Fantasy</th>
      <th>Romance</th>
      <th>...</th>
      <th>Horror</th>
      <th>Mystery</th>
      <th>Sci-Fi</th>
      <th>IMAX</th>
      <th>Documentary</th>
      <th>War</th>
      <th>Musical</th>
      <th>Western</th>
      <th>Film-Noir</th>
      <th>(no genres listed)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story</td>
      <td>[Adventure, Animation, Children, Comedy, Fantasy]</td>
      <td>1995</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji</td>
      <td>[Adventure, Children, Fantasy]</td>
      <td>1995</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>293</th>
      <td>296</td>
      <td>Pulp Fiction</td>
      <td>[Comedy, Crime, Drama, Thriller]</td>
      <td>1994</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1246</th>
      <td>1274</td>
      <td>Akira</td>
      <td>[Action, Adventure, Animation, Sci-Fi]</td>
      <td>1988</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1885</th>
      <td>1968</td>
      <td>Breakfast Club, The</td>
      <td>[Comedy, Drama]</td>
      <td>1985</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>



We'll only need the actual genre table, so let's clean this up a bit by resetting the index and dropping the __movieId, title, genres__ and __year__ columns.


```python
#Resetting the index to avoid future issues
user_movies = user_movies.reset_index(drop=True)
#Dropping unnecessary issues due to save memory and to avoid issues
userGenreTable = user_movies.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
userGenreTable
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Adventure</th>
      <th>Animation</th>
      <th>Children</th>
      <th>Comedy</th>
      <th>Fantasy</th>
      <th>Romance</th>
      <th>Drama</th>
      <th>Action</th>
      <th>Crime</th>
      <th>Thriller</th>
      <th>Horror</th>
      <th>Mystery</th>
      <th>Sci-Fi</th>
      <th>IMAX</th>
      <th>Documentary</th>
      <th>War</th>
      <th>Musical</th>
      <th>Western</th>
      <th>Film-Noir</th>
      <th>(no genres listed)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



Now we're ready to start learning the input's preferences!

To do this, we're going to turn each genre into weights. We can do this by using the input's reviews and multiplying them into the input's genre table and then summing up the resulting table by column. This operation is actually a dot product between a matrix and a vector, so we can simply accomplish by calling Pandas's "dot" function.


```python
input_movies['rating']
```




    0    3.5
    1    2.0
    2    5.0
    3    4.5
    4    5.0
    Name: rating, dtype: float64




```python
#Dot produt to get weights
userProfile = userGenreTable.transpose().dot(input_movies['rating'])
#The user profile
userProfile
```




    Adventure             10.0
    Animation              8.0
    Children               5.5
    Comedy                13.5
    Fantasy                5.5
    Romance                0.0
    Drama                 10.0
    Action                 4.5
    Crime                  5.0
    Thriller               5.0
    Horror                 0.0
    Mystery                0.0
    Sci-Fi                 4.5
    IMAX                   0.0
    Documentary            0.0
    War                    0.0
    Musical                0.0
    Western                0.0
    Film-Noir              0.0
    (no genres listed)     0.0
    dtype: float64



Now, we have the weights for every of the user's preferences. This is known as the User Profile ( __userProfile__ ). Using this, we can recommend movies that satisfy the user's preferences.

Let's start by extracting the genre table from the original dataframe:


```python
#Now let's get the genres of every movie in our original dataframe
genreTable = df_movgen.set_index(df_movgen['movieId'])
#And drop the unnecessary information
genreTable = genreTable.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
genreTable.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Adventure</th>
      <th>Animation</th>
      <th>Children</th>
      <th>Comedy</th>
      <th>Fantasy</th>
      <th>Romance</th>
      <th>Drama</th>
      <th>Action</th>
      <th>Crime</th>
      <th>Thriller</th>
      <th>Horror</th>
      <th>Mystery</th>
      <th>Sci-Fi</th>
      <th>IMAX</th>
      <th>Documentary</th>
      <th>War</th>
      <th>Musical</th>
      <th>Western</th>
      <th>Film-Noir</th>
      <th>(no genres listed)</th>
    </tr>
    <tr>
      <th>movieId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
genreTable.shape
```




    (34208, 20)




```python
#Multiply the genres by the weights and then take the weighted average
recommendationTable_df = ((genreTable*userProfile).sum(axis=1))/(userProfile.sum())
recommendationTable_df.head()
```




    movieId
    1    0.594406
    2    0.293706
    3    0.188811
    4    0.328671
    5    0.188811
    dtype: float64



With the input's profile and the complete list of movies and their genres in hand, we're going to take the weighted average of every movie based on the input profile and recommend the top 5 movies that most satisfy it.


```python
#Sort our recommendations in descending order
recommendationTable_df = recommendationTable_df.sort_values(ascending=False)
#Just a peek at the values
recommendationTable_df.head()
```




    movieId
    5018      0.748252
    26093     0.734266
    27344     0.720280
    148775    0.685315
    6902      0.678322
    dtype: float64




```python
#The final recommendation table
x=recommendationTable_df.head(5).keys()
## df_movie.query('movieId in @x')
## df_movie[df_movie['movieId'].isin(recommendationTable_df.head(5).keys())]
##### BY ORDER
mov_byitem=df_movie.copy()
mov_byitem=mov_byitem.set_index('movieId').loc[x].reset_index(inplace=False)
```


```python
mov_byitem
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5018</td>
      <td>Motorama</td>
      <td>[Adventure, Comedy, Crime, Drama, Fantasy, Mys...</td>
      <td>1991</td>
    </tr>
    <tr>
      <th>1</th>
      <td>26093</td>
      <td>Wonderful World of the Brothers Grimm, The</td>
      <td>[Adventure, Animation, Children, Comedy, Drama...</td>
      <td>1962</td>
    </tr>
    <tr>
      <th>2</th>
      <td>27344</td>
      <td>Revolutionary Girl Utena: Adolescence of Utena...</td>
      <td>[Action, Adventure, Animation, Comedy, Drama, ...</td>
      <td>1999</td>
    </tr>
    <tr>
      <th>3</th>
      <td>148775</td>
      <td>Wizards of Waverly Place: The Movie</td>
      <td>[Adventure, Children, Comedy, Drama, Fantasy, ...</td>
      <td>2009</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6902</td>
      <td>Interstate 60</td>
      <td>[Adventure, Comedy, Drama, Fantasy, Mystery, S...</td>
      <td>2002</td>
    </tr>
  </tbody>
</table>
</div>




```python
rate=df_rating[df_rating['movieId'].isin(mov_byitem['movieId'].tolist())].groupby('movieId')['rating'].sum()/df_rating[df_rating['movieId'].isin(mov_byitem['movieId'].tolist())].groupby('movieId')['rating'].count()
```


```python
rate_pd=pd.DataFrame(rate)
```


```python
rate_pd.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rating</th>
    </tr>
    <tr>
      <th>movieId</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5018</th>
      <td>3.130435</td>
    </tr>
    <tr>
      <th>6902</th>
      <td>3.866979</td>
    </tr>
    <tr>
      <th>26093</th>
      <td>3.500000</td>
    </tr>
    <tr>
      <th>27344</th>
      <td>3.333333</td>
    </tr>
    <tr>
      <th>148775</th>
      <td>3.166667</td>
    </tr>
  </tbody>
</table>
</div>




```python
mov_byitem=mov_byitem.merge(rate_pd, left_on='movieId', right_on='movieId')
```


```python
print('MOVIES YOU MIGHT LIKE')
mov_byitem
```

    MOVIES YOU MIGHT LIKE





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>year</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5018</td>
      <td>Motorama</td>
      <td>[Adventure, Comedy, Crime, Drama, Fantasy, Mys...</td>
      <td>1991</td>
      <td>3.130435</td>
    </tr>
    <tr>
      <th>1</th>
      <td>26093</td>
      <td>Wonderful World of the Brothers Grimm, The</td>
      <td>[Adventure, Animation, Children, Comedy, Drama...</td>
      <td>1962</td>
      <td>3.500000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>27344</td>
      <td>Revolutionary Girl Utena: Adolescence of Utena...</td>
      <td>[Action, Adventure, Animation, Comedy, Drama, ...</td>
      <td>1999</td>
      <td>3.333333</td>
    </tr>
    <tr>
      <th>3</th>
      <td>148775</td>
      <td>Wizards of Waverly Place: The Movie</td>
      <td>[Adventure, Children, Comedy, Drama, Fantasy, ...</td>
      <td>2009</td>
      <td>3.166667</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6902</td>
      <td>Interstate 60</td>
      <td>[Adventure, Comedy, Drama, Fantasy, Mystery, S...</td>
      <td>2002</td>
      <td>3.866979</td>
    </tr>
  </tbody>
</table>
</div>



# User based filter

attempts to find users that have similar preferences and opinions as the input and then recommends items that they have liked to the input. We will be using a method based on the __Pearson Correlation Function__.


The process for creating a User Based recommendation system is as follows:
- Select a user with the movies the user has watched
- Based on his rating to movies, find the top X neighbours 
- Get the watched movie record of the user for each neighbour.
- Calculate a similarity score using some formula
- Recommend the items with the highest score


Let's begin by creating an input user to recommend movies to:


```python
userInput = [
            {'title':'Breakfast Club, The', 'rating':5},
            {'title':'Toy Story', 'rating':3.5},
            {'title':'Jumanji', 'rating':2},
            {'title':"Pulp Fiction", 'rating':5},
            {'title':'Akira', 'rating':4.5}
         ] 
inputMovies = pd.DataFrame(userInput)
```

Add movieId to inputMovies



```python
#Filtering out the movies by title
inputId = df_movie[df_movie['title'].isin(inputMovies['title'].tolist())]
#Then merging it so we can get the movieId. It's implicitly merging it by title.
inputMovies = pd.merge(inputId, inputMovies)
#Dropping information we won't use from the input dataframe
inputMovies = inputMovies.drop('year', 1)
inputMovies = inputMovies.drop('genres', 1)
#Final input dataframe
#If a movie you added in above isn't here, then it might not be in the original 
#dataframe or it might spelled differently, please check capitalisation.
inputMovies
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>296</td>
      <td>Pulp Fiction</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1274</td>
      <td>Akira</td>
      <td>4.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1968</td>
      <td>Breakfast Club, The</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>



Now with the movie ID's in our input, we can now get the subset of users that have watched and reviewed the movies in our input.




```python
#Filtering out users that have watched movies that the input has watched and storing it
userSubset = df_rating[df_rating['movieId'].isin(inputMovies['movieId'].tolist())]
userSubset.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19</th>
      <td>4</td>
      <td>296</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>441</th>
      <td>12</td>
      <td>1968</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>479</th>
      <td>13</td>
      <td>2</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>531</th>
      <td>13</td>
      <td>1274</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>681</th>
      <td>14</td>
      <td>296</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>



We now group up the rows by user ID.


```python
userSubsetGroup = userSubset.groupby(['userId'])
```

Lets look at one of the users, e.g. the one with userID=1607


```python
len(userSubsetGroup)
```




    116140



check some row


```python
userSubsetGroup.get_group(1607)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>144611</th>
      <td>1607</td>
      <td>1</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>144612</th>
      <td>1607</td>
      <td>2</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>144664</th>
      <td>1607</td>
      <td>296</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>145034</th>
      <td>1607</td>
      <td>1968</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>



Let's also sort these groups so the users that share the most movies in common with the input have higher priority. This provides a richer recommendation since we won't go through every single user.


```python
userSubsetGroup = sorted(userSubsetGroup,  key=lambda x: len(x[1]), reverse=True)
```

Now lets look at the first 5 users


```python
userSubsetGroup[0:5]
```




    [(75,       userId  movieId  rating
      7507      75        1     5.0
      7508      75        2     3.5
      7540      75      296     5.0
      7633      75     1274     4.5
      7673      75     1968     5.0), (106,       userId  movieId  rating
      9083     106        1     2.5
      9084     106        2     3.0
      9115     106      296     3.5
      9198     106     1274     3.0
      9238     106     1968     3.5), (686,        userId  movieId  rating
      61336     686        1     4.0
      61337     686        2     3.0
      61377     686      296     4.0
      61478     686     1274     4.0
      61569     686     1968     5.0), (815,        userId  movieId  rating
      73747     815        1     4.5
      73748     815        2     3.0
      73922     815      296     5.0
      74362     815     1274     3.0
      74678     815     1968     4.5), (1040,        userId  movieId  rating
      96689    1040        1     3.0
      96690    1040        2     1.5
      96733    1040      296     3.5
      96859    1040     1274     3.0
      96922    1040     1968     4.0)]



#### Similarity of users to input user
Next, we are going to compare some users to our specified user and find the one that is most similar.  
we're going to find out how similar each user is to the input through the __Pearson Correlation Coefficient__. It is used to measure the strength of a linear association between two variables.

![alt text](https://wikimedia.org/api/rest_v1/media/math/render/svg/bd1ccc2979b0fd1c1aec96e386f686ae874f9ec0 "Pearson Correlation")

In our case, a 1 means that the two users have similar tastes while a -1 means the opposite.

We will select a subset of users to iterate through. This limit is imposed because we don't want to waste too much time going through every single user.


```python
userSubsetGroup = userSubsetGroup[0:50000]
```

Now, we calculate the Pearson Correlation between input user and subset group, and store it in a dictionary




```python
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


```


```python
pearsonDF = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')
pearsonDF.columns = ['similarityIndex']
pearsonDF['userId'] = pearsonDF.index
pearsonDF.index = range(len(pearsonDF))
pearsonDF.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>similarityIndex</th>
      <th>userId</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.827278</td>
      <td>75</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.586009</td>
      <td>106</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.832050</td>
      <td>686</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.576557</td>
      <td>815</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.943456</td>
      <td>1040</td>
    </tr>
  </tbody>
</table>
</div>




Now let's get the top 50 users that are most similar to the input.


```python
topUsers=pearsonDF.sort_values(by='similarityIndex', ascending=False)[0:50]
topUsers.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>similarityIndex</th>
      <th>userId</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>18014</th>
      <td>1.0</td>
      <td>191754</td>
    </tr>
    <tr>
      <th>14426</th>
      <td>1.0</td>
      <td>128007</td>
    </tr>
    <tr>
      <th>10574</th>
      <td>1.0</td>
      <td>57589</td>
    </tr>
    <tr>
      <th>19024</th>
      <td>1.0</td>
      <td>209126</td>
    </tr>
    <tr>
      <th>16805</th>
      <td>1.0</td>
      <td>170013</td>
    </tr>
  </tbody>
</table>
</div>



Now, let's start recommending movies to the input user.

We're going to do this by taking the weighted average of the ratings of the movies using the Pearson Correlation as the weight. But to do this, we first need to get the movies watched by the users in our __pearsonDF__ from the ratings dataframe and then store their correlation in a new column called _similarityIndex_. This is achieved below by merging of these two tables.


```python
topUsersRating=topUsers.merge(df_rating, left_on='userId', right_on='userId', how='inner')
topUsersRating.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>similarityIndex</th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>191754</td>
      <td>1</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>191754</td>
      <td>6</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>191754</td>
      <td>10</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>191754</td>
      <td>16</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>191754</td>
      <td>47</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>



Now all we need to do is simply multiply the movie rating by its weight (The similarity index), then sum up the new ratings and divide it by the sum of the weights.

We can easily do this by simply multiplying two columns, then grouping up the dataframe by movieId and then dividing two columns:

It shows the idea of all similar users to candidate movies for the input user:


```python
#Multiplies the similarity by the user's ratings
topUsersRating['weightedRating'] = topUsersRating['similarityIndex']*topUsersRating['rating']
topUsersRating.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>similarityIndex</th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>weightedRating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>191754</td>
      <td>1</td>
      <td>3.5</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>191754</td>
      <td>6</td>
      <td>3.5</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>191754</td>
      <td>10</td>
      <td>3.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>191754</td>
      <td>16</td>
      <td>4.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>191754</td>
      <td>47</td>
      <td>4.0</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Applies a sum to the topUsers after grouping it up by userId
tempTopUsersRating = topUsersRating.groupby('movieId').sum()[['similarityIndex','weightedRating']]
tempTopUsersRating.columns = ['sum_similarityIndex','sum_weightedRating']
tempTopUsersRating.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sum_similarityIndex</th>
      <th>sum_weightedRating</th>
    </tr>
    <tr>
      <th>movieId</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>48.0</td>
      <td>175.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>7.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.0</td>
      <td>7.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5.0</td>
      <td>12.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Creates an empty dataframe
df_rec = pd.DataFrame()
#Now we take the weighted average
df_rec['weighted average recommendation score'] = tempTopUsersRating['sum_weightedRating']/tempTopUsersRating['sum_similarityIndex']
# df_rec['movieId'] = tempTopUsersRating.index
df_rec.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>weighted average recommendation score</th>
    </tr>
    <tr>
      <th>movieId</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>3.645833</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.750000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.750000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2.400000</td>
    </tr>
  </tbody>
</table>
</div>



Now let's sort it and see the top 20 movies that the algorithm recommended!


```python
df_rec = df_rec.sort_values(by='weighted average recommendation score', ascending=False)
df_rec.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>weighted average recommendation score</th>
    </tr>
    <tr>
      <th>movieId</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>85394</th>
      <td>5.0</td>
    </tr>
    <tr>
      <th>71926</th>
      <td>5.0</td>
    </tr>
    <tr>
      <th>51935</th>
      <td>5.0</td>
    </tr>
    <tr>
      <th>6477</th>
      <td>5.0</td>
    </tr>
    <tr>
      <th>9005</th>
      <td>5.0</td>
    </tr>
    <tr>
      <th>60333</th>
      <td>5.0</td>
    </tr>
    <tr>
      <th>5066</th>
      <td>5.0</td>
    </tr>
    <tr>
      <th>71899</th>
      <td>5.0</td>
    </tr>
    <tr>
      <th>3645</th>
      <td>5.0</td>
    </tr>
    <tr>
      <th>93040</th>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# df_movie.loc[df_movie['movieId'].isin(df_rec.head(10)['movieId'].tolist())]
## BY ORDER
# xy=df_rec.head(5).index.tolist()
# df_mvx=df_movie.copy()
# df_mvx=df_mvx[df_mvx['movieId'].isin(xy)]
```


```python
df_mvu=df_movie.copy()
df_mvu=df_rec.merge(df_mvu,left_on='movieId',right_on='movieId',how='inner')
df_mvu.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>weighted average recommendation score</th>
      <th>title</th>
      <th>genres</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>85394</td>
      <td>5.0</td>
      <td>Cave of Forgotten Dreams</td>
      <td>[Documentary]</td>
      <td>2010</td>
    </tr>
    <tr>
      <th>1</th>
      <td>71926</td>
      <td>5.0</td>
      <td>Boys Are Back, The</td>
      <td>[Comedy, Drama]</td>
      <td>2009</td>
    </tr>
    <tr>
      <th>2</th>
      <td>51935</td>
      <td>5.0</td>
      <td>Shooter</td>
      <td>[Action, Drama, Thriller]</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6477</td>
      <td>5.0</td>
      <td>Song of Bernadette, The</td>
      <td>[Drama]</td>
      <td>1943</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9005</td>
      <td>5.0</td>
      <td>Fire in the Sky</td>
      <td>[Drama, Mystery, Sci-Fi]</td>
      <td>1993</td>
    </tr>
  </tbody>
</table>
</div>




```python
rat=df_rating[df_rating['movieId'].isin(df_mvu['movieId'].tolist())].groupby('movieId')['rating'].sum()/df_rating[df_rating['movieId'].isin(df_mvu['movieId'].tolist())].groupby('movieId')['rating'].count()
```


```python
rate_us=pd.DataFrame(rat)
```


```python
df_mvu=df_mvu.merge(rate_us,left_on='movieId',right_on='movieId')
```


```python
df_mvu_new=df_mvu.sort_values(by=['weighted average recommendation score','rating'],ascending=[False,False]).reset_index(inplace=False)
df_mvu_new=df_mvu_new.drop(['index'],axis=1)
```


```python
df_mvu_rec=df_mvu_new.loc[0:5,('movieId','title','genres','year','rating')]
print('Other people who watched the movie(s) also watched and liked')
df_mvu_rec.head()
```

    Other people who watched the movie(s) also watched and liked





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>year</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>109571</td>
      <td>Into the Middle of Nowhere</td>
      <td>[Adventure, Children, Comedy, Documentary, Drama]</td>
      <td>2010</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>109715</td>
      <td>Inquire Within</td>
      <td>[Comedy]</td>
      <td>2012</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>109598</td>
      <td>Vertical Features Remake</td>
      <td>[Documentary]</td>
      <td>1978</td>
      <td>4.250000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>116897</td>
      <td>Wild Tales</td>
      <td>[Comedy, Drama, Thriller]</td>
      <td>2014</td>
      <td>4.107895</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7327</td>
      <td>Persona</td>
      <td>[Drama]</td>
      <td>1966</td>
      <td>4.097098</td>
    </tr>
  </tbody>
</table>
</div>



Conclusion:
Item based technique is higliy personalised for the user, but it does not take into account what others think of the item. User based technique takes other user's ratings into consideration and adapts to the user's interests which might change over time, but it may takes a longer time to process the system

