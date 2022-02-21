import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame 
import nltk 
import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors
from scipy.spatial.distance import cosine
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
import re
import string
from wordcloud import WordCloud, STOPWORDS
# from sklearn.metrics import mean_squared_error

df = pd.read_json("/recom/data_amazon.json",lines=True)
df[['Helpful','Total_whethernot_helpful']] = pd.DataFrame(df.helpful.values.tolist(), index = df.index)
df.drop_duplicates(subset=['reviewerID', 'asin','unixReviewTime'],inplace=True)
df['Helpful %'] = np.where(df['Total_whethernot_helpful'] > 0, df['Helpful'] / df['Total_whethernot_helpful'], -1)
df['% Upvote'] = pd.cut(df['Helpful %'], bins = [-1, 0, 0.2, 0.4, 0.6, 0.8, 1.0], labels = ['Empty', '0-20%', '20-40%', '40-60%', '60-80%', '80-100%'], include_lowest = True)
df['Id'] = df.index;
df["usefulScore"] = (df["Helpful"]/df["Total_whethernot_helpful"]).apply(lambda n: "useful" if n > 0.8 else "useless")
df=df.dropna(subset=['% Upvote'])
# print(df.columns)
# print(df.shape)
count = df.groupby("asin", as_index=False).count()
# mean = df.groupby("asin", as_index=False).mean()
dfMerged = pd.merge(df, count, how='right', on=['asin'])
# print(dfMerged.head())
#rename column
dfMerged["totalReviewers"] = dfMerged["reviewerID_y"]
dfMerged["overallScore"] = dfMerged["overall_x"]
dfMerged["summaryReview"] = dfMerged["summary_x"]
# dfNew = dfMerged[['asin','summaryReview','overallScore',"totalReviewers"]]

# Selecting products which have more than 80 reviews
dfMerged = dfMerged.sort_values(by='totalReviewers', ascending=False)
dfCount = dfMerged[dfMerged.totalReviewers >= 80]
# print(dfCount.head())
# print(dfCount.shape)
# print(dfCount['totalReviewers'].describe())

# ### Grouping all the summary Reviews by product ID
dfProductReview = df.groupby("asin", as_index=False).mean()
ProductReviewSummary = dfCount.groupby("asin")["summaryReview"].apply(str)
ProductReviewSummary = pd.DataFrame(ProductReviewSummary)
# print(ProductReviewSummary.head())

# print(dfProductReview.shape)
# print(ProductReviewSummary.shape)
# print(dfProductReview.head())

# ### create dataframe with certain columns
df3 = pd.merge(ProductReviewSummary, dfProductReview, on="asin", how='inner')
df3 = df3[['asin','summaryReview','overall']]
# ### Text Cleaning - Summary column
# print(df3.head())
#function for tokenizing summary
regEx = re.compile('[^a-zA-Z]+')
def cleanReviews(reviewText):
    reviewText = regEx.sub(' ', reviewText).strip()
    return reviewText
#reset index and drop duplicate rows
df3["summaryClean"] = df3["summaryReview"].apply(cleanReviews)
# df3 = df3.drop_duplicates(['overall'], keep='last')
# df3 = df3.reset_index()
reviews = df3["summaryClean"] 
countVector = CountVectorizer(max_features = 300, stop_words='english') 
transformedReviews = countVector.fit_transform(reviews) 
pickle.dump(countVector, open('pro_countv.pkl','wb'))
dfReviews = DataFrame(transformedReviews.A, columns=countVector.get_feature_names())
dfReviews = dfReviews.astype(int)

# import random
# First let's create a dataset called X
X = np.array(dfReviews)
# create train and test
tpercent = 0.85
np.random.seed(9)
tsize = int(np.floor(tpercent * len(dfReviews)))
i_x=np.random.rand(len(dfReviews))<tpercent
dfReviews_train = X[i_x]
dfReviews_test = X[~i_x]
# len of train and test
lentrain = len(dfReviews_train)
lentest = len(dfReviews_test)
# KNN classifier to find similar products
# print(lentrain)
# print(lentest)

neighbor = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(dfReviews_train)
# Let's find the k-neighbors of each point in object X. To do that we call the kneighbors() function on object X.
# distances, indices = neighbor.kneighbors(dfReviews_train)

#find most related products
# m=0
# for i in ~i_x:
#     if i==True:
#         a = neighbor.kneighbors([X[m]])
#         related_product_list = a[1] #index of product we bought
#         first_related_product = [item[0] for item in related_product_list]
#         # print(first_related_product)
#         first_related_product = str(first_related_product).strip('[]')
#         first_related_product = int(first_related_product)
#         second_related_product = [item[1] for item in related_product_list]
#         second_related_product = str(second_related_product).strip('[]')
#         second_related_product = int(second_related_product)
#         # print ("Based on product reviews, for ", df3["asin"][m] ," average rating is ",df3["overall"][m])
#         # print ("The first similar product is ", df3["asin"][first_related_product] ," average rating is ",df3["overall"][first_related_product])
#         # print ("The second similar product is ", df3["asin"][second_related_product] ," average rating is ",df3["overall"][second_related_product])
#         # print ("-----------------------------------------------------------")
#     m+=1
pickle.dump(neighbor, open('pro_rec.pkl','wb'))
# pickle.dump(cv,open('news_cv.pkl','wb'))
# pickle.dump(encoder,open('enc_news.pkl','wb'))
# print ("Based on product reviews, for ", df3["asin"][117] ," average rating is ",df3["overall"][117])
# print ("The first similar product is ", df3["asin"][first_related_product] ," average rating is ",df3["overall"][first_related_product])
# print ("The second similar product is ", df3["asin"][second_related_product] ," average rating is ",df3["overall"][second_related_product])
# print ("-----------------------------------------------------------")

# ### Predicting Review Score
# df5_train_target = df3["overall"][i_x]
# df5_test_target = df3["overall"][~i_x]
# df5_train_target = df5_train_target.astype(int)
# df5_test_target = df5_test_target.astype(int)

# n_neighbors = 3
# knnclf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
# knnclf.fit(dfReviews_train, df5_train_target)
# knnpreds_test = knnclf.predict(dfReviews_test)

# print(classification_report(df5_test_target, knnpreds_test))
# print (accuracy_score(df5_test_target, knnpreds_test))
# print(mean_squared_error(df5_test_target, knnpreds_test))

# ### Predicting Review Score with k = rule of thumb
# if round(lentrain**0.5)%2==0:
#     co=round(lentrain**0.5)+1
# else:
#     co=round(lentrain**0.5)

# knncl5 = neighbors.KNeighborsClassifier(co, weights='distance')
# knncl5.fit(dfReviews_train, df5_train_target)
# knnpred5_test = knncl5.predict(dfReviews_test)
# print(classification_report(df5_test_target, knnpred5_test))
# print (accuracy_score(df5_test_target, knnpred5_test))
# print(mean_squared_error(df5_test_target, knnpred5_test))


df['newsumma']=df["summary"].apply(cleanReviews)
df_inp =  df[df['overall'] != 3]
X_inp = df_inp['newsumma']
y_dict = {1:0, 2:0, 4:1, 5:1} # 0 = negative ; 1 = positive
y_inp = df_inp['overall'].map(y_dict)
Xtrain, Xtest, ytrain,ytest = train_test_split(X_inp,y_inp, test_size=0.2,random_state=3)
countVec = CountVectorizer(stop_words='english',min_df = 1, ngram_range = (1, 4)) #min_df ignore words that appear in less than 1 document and ngram_range sets min_n&max_n for a sequence of N words
X_train_vec = countVec.fit_transform(Xtrain)
pickle.dump(countVec, open('pro_crat.pkl','wb'))
tfidf_transformer = TfidfTransformer()
X_tra_tfidf = tfidf_transformer.fit_transform(X_train_vec)
pickle.dump(tfidf_transformer, open('pro_tfidrat.pkl','wb'))
X_te_vec = countVec.transform(Xtest)
X_te_tfidf = tfidf_transformer.transform(X_te_vec)

logreg = LogisticRegression(C=1e5) #sronger regularization
logreg.fit(X_tra_tfidf, ytrain)
pickle.dump(logreg, open('pro_lograt.pkl','wb'))

###############################

#Read the file and add new columns helpfulnessnumerator and helpfulnessdenominator, so we can measure how helpful the review is
# reviews = pd.read_json('data_amazon.json',lines=True)
# reviews[['Helpful','Total_whethernot_helpful']] = pd.DataFrame(reviews.helpful.values.tolist(), index = reviews.index)
# reviews.shape
# reviews.head()

#Cleaning the data by eliminating duplicates
# reviews.drop_duplicates(subset=['reviewerID', 'asin','unixReviewTime'],inplace=True)

#Adding the helpfulness and upvote percentages for metrics
# reviews['Helpful %'] = np.where(reviews['Total_whethernot_helpful'] > 0, reviews['Helpful'] / reviews['Total_whethernot_helpful'], -1)
# reviews['% Upvote'] = pd.cut(reviews['Helpful %'], bins = [-1, 0, 0.2, 0.4, 0.6, 0.8, 1.0], labels = ['Empty', '0-20%', '20-40%', '40-60%', '60-80%', '80-100%'], include_lowest = True)
# reviews['Id'] = reviews.index;

# reviews["usefulScore"] = (reviews["Helpful"]/reviews["Total_whethernot_helpful"]).apply(lambda n: "useful" if n > 0.8 else "useless")


# df_s = reviews.groupby(['overall', '% Upvote']).agg({'Id': 'count'})
# df_s = df_s.unstack()
# df_s.columns = df_s.columns.get_level_values(1)
# df_s

# regEx = re.compile('[^a-z]+') #besides alphabet
# def cleanReviews(reviewText):
#     reviewText = reviewText.lower()
#     reviewText = regEx.sub(' ', reviewText).strip() #replace regEX with space ' ' and remove spaces at the begining and end of the review
#     return reviewText
# reviews['newsumma']=reviews["summary"].apply(cleanReviews)
# df =  reviews[reviews['overall'] != 3]
# X = df['newsumma']
# y_dict = {1:0, 2:0, 4:1, 5:1} # 0 = negative ; 1 = positive
# y = df['overall'].map(y_dict)
# y.value_counts()

# __check if theres any missing value__
# reviews.isna().sum()

# We need to remove a row that has Helpful is less than Total_whethernot_helpful
# reviews=reviews.dropna(subset=['% Upvote'])
# reviews[reviews['% Upvote'].isna()]

# __displaying most frequently used words in summary__
# from wordcloud import WordCloud, STOPWORDS
# stopwords = set(STOPWORDS)
# def show_wordcloud(data, title = None):
#     wordcloud = WordCloud(
#         background_color='white',
#         stopwords=stopwords,
#         max_words=80,
#         max_font_size=40, 
#         random_state=1 
#     ).generate(str(data))
#     fig = plt.figure(1, figsize=(8, 8))
#     plt.axis('off')
#     if title: 
#         fig.suptitle(title, fontsize=20)
#         fig.subplots_adjust(top=1.3)
#     plt.imshow(wordcloud)
#     plt.show()
# show_wordcloud(reviews['newsumma'],title='Top 80 Words frequenly used')
# show_wordcloud(reviews['newsumma'][reviews['overall'].isin([1,2])],'Top 80 Words frequenly used for negative summary')
# show_wordcloud(reviews['newsumma'][reviews['overall'].isin([4,5])],'Top 80 Words frequenly used for positive summary')

# Xtrain, Xtest, ytrain,ytest = train_test_split(X,y, test_size=0.2,random_state=3)
# print("%d items in training data, %d in test data" % (len(Xtrain), len(Xtest)))


# countVector = CountVectorizer(stop_words='english',min_df = 1, ngram_range = (1, 4)) #min_df ignore words that appear in less than 1 document and ngram_range sets min_n&max_n for a sequence of N words
# X_train_counts = countVector.fit_transform(Xtrain)

#applying tfidf to term frequency
# tfidf_transformer = TfidfTransformer()
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# X_new_counts = countVector.transform(Xtest)
# X_test_tfidf = tfidf_transformer.transform(X_new_counts)

# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)

# ## Logistic Reg
# print(X_train_tfidf.shape)

# from sklearn.linear_model import LogisticRegression
# prediction=dict()
# logreg = LogisticRegression(C=1e5) #sronger regularization
# logreg_result = logreg.fit(X_train_tfidf, ytrain)