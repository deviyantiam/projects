# The project is to identify topic of news

# !pip install PySastrawi
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
# from sklearn.model_selection import RandomizedSearchCV
import requests
import pickle


# # # Data Preparation
# # prepare the dataset by gathering and checking if theres any missing value and remove them
# # ## Data Gathering
# ## collecting data
data_csv=pd.read_csv('data.csv')
# print('Shape of dataset =', data_csv.shape)
# print(data_csv.head(5))

# # ## Data Cleaning
# ## check missing value
# print(data_csv.isnull().sum())

# ## check any cells that have missing value
# print(data_csv[data_csv['article_content'].isna()])

# ## drop those cells
data_csv=data_csv.dropna(how='any')
# print(data_csv.isna().sum())

# ## find unique values from 'article_topic'
topi=data_csv['article_topic'].unique().tolist()
# print('List of topics:')
# print(*topi, sep=', ')

# ## class distribution
# for p in list(set(data_csv.article_topic)):
#     print('number of',p,' ',len(data_csv.loc[data_csv['article_topic'] == p]))

# ## convert categorical values into numeric values
encoder = LabelEncoder()
encoder.fit(data_csv['article_topic'])
data_csv['index_topic']= encoder.transform(data_csv['article_topic'])
# print(data_csv.head())


# # # Feature Engineering
# # raw text of training and test data will be transformed into new feature by going through some processes below. Stopword removing is to remove the most common words in a language and stemming is to reduce a word by remove its affixes. After re-runing some process multiple times, we noticed there were rows that had the same article contents. Therefore, the latest version of the script for stemming and stopword removing was created as follows:

## import StemmerFactory class
factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()

# ## create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# ## create new column of stemmed and stopwords removed articles
a_content=data_csv.article_content

for i in a_content:
    stops=stopword.remove(stemmer.stem(i))
    wordy=''
    for st in stops.split(' '):
        if st.isalpha():      
            wordy+=st+' '
    # dealing with same articles
    indx=data_csv.loc[data_csv['article_content']==i].index.tolist()
    data_csv.loc[indx,'article_new']=wordy


# #### After multiple process of trial and error, I found that there were a few duplicate articles, eg below shows that row 791 and 107 were the same. That's why i put this command in cell above
# ``` pyhton
#     indx=data_csv.loc[data_csv['article_content']==i].index.tolist()
#     data_csv.loc[indx,'article_new']=wordy
# ```

# data_csv.loc[data_csv['article_content']==data_csv['article_content'][791]]
# data_csv.loc[data_csv['article_content']==data_csv['article_content'][791]].index[0]
# data_csv.loc[data_csv['article_content']==data_csv['article_content'][791]].index[1]

# ## splitting
# X_train, X_test, y_train, y_test = train_test_split(data_csv['article_new'], data_csv['index_topic'], test_size=.15, random_state = 79)
# print("Training dataset: ", X_train.shape[0])
# print("Test dataset: ", X_test.shape[0])


# ```python
# ##  instantiate CountVectorizer()
# cv = CountVectorizer()
#  
# ## this steps generates word counts for the words in your dataset
# x_word_count_vector = cv.fit_transform(X_train)
# ```
# #### After splitting, we ran the cell above and we got an error message: 
# ```python
# ---------------------------------------------------------------------------
# AttributeError                            Traceback (most recent call last)
# <ipython-input-13-6036975bf677> in <module>
#       3 
#       4 ## this steps generates word counts for the words in your dataset
# ----> 5 x_word_count_vector = cv.fit_transform(X_train)
#       6 ## word_count_vector
# 
# ~/anaconda3/lib/python3.7/site-packages/sklearn/feature_extraction/text.py in fit_transform(self, raw_documents, y)
#    1029 
#    1030         vocabulary, X = self._count_vocab(raw_documents,
# -> 1031                                           self.fixed_vocabulary_)
#    1032 
#    1033         if self.binary:
# 
# ~/anaconda3/lib/python3.7/site-packages/sklearn/feature_extraction/text.py in _count_vocab(self, raw_documents, fixed_vocab)
#     941         for doc in raw_documents:
#     942             feature_counter = {}
# --> 943             for feature in analyze(doc):
#     944                 try:
#     945                     feature_idx = vocabulary[feature]
# 
# ~/anaconda3/lib/python3.7/site-packages/sklearn/feature_extraction/text.py in <lambda>(doc)
#     327                                                tokenize)
#     328             return lambda doc: self._word_ngrams(
# --> 329                 tokenize(preprocess(self.decode(doc))), stop_words)
#     330 
#     331         else:
# 
# ~/anaconda3/lib/python3.7/site-packages/sklearn/feature_extraction/text.py in <lambda>(x)
#     255 
#     256         if self.lowercase:
# --> 257             return lambda x: strip_accents(x.lower())
#     258         else:
#     259             return strip_accents
# 
# AttributeError: 'float' object has no attribute 'lower'
# ```
# #### it turned out there were missing values in some rows, we would analize how it could be possible. We already removed the missing value though. Perhaps, we would track down where the problem was

# ## check if there's any string cell contains blank space
id_=data_csv[data_csv['article_new']==''].index.tolist()
# print(data_csv[data_csv['article_new']==''])

# ## replace blank space with np.nan
for i in id_:
    data_csv.loc[i] = data_csv.loc[i].replace('',np.nan)

# print(data_csv[data_csv['article_new'].isnull()])
# print(data_csv.isnull().sum())


# #### There were rows whose 'article_content' only contained spaces '  '. 
# print(data_csv.article_content[9261])

# ## drop those cells
data_new=data_csv.dropna(how='any')
# print(data_new.isna().sum())

# ### Split data to training and test data
X_train, X_test, y_train, y_test = train_test_split(data_new['article_new'], data_new['index_topic'], test_size=.15, random_state = 89)
# print("Training dataset: ", X_train.shape[0])
# print("Test dataset: ", X_test.shape[0])

# ##  instantiate CountVectorizer()
cv = CountVectorizer()
 
# ## this steps generates word counts for the words in your dataset
x_word_count_vector = cv.fit_transform(X_train)
x_testing_count = cv.transform(X_test)


# # Model Building
# na_bayes = MultinomialNB()
# na_bayes.fit(x_word_count_vector, y_train)
# pred_nb = na_bayes.predict(x_testing_count)

# sgdc_=SGDClassifier(random_state=42)
# sgdc_.fit(x_word_count_vector, y_train)
# pred_sgdc = sgdc_.predict(x_testing_count)


# ## EVALUATION
# print('Accurary of Naive Bayes:', accuracy_score(y_test,pred_nb))
# print(classification_report(y_test, pred_nb, target_names=encoder.classes_))

# print('Accurary of SGDC:', accuracy_score(y_test,pred_sgdc))
# print(classification_report(y_test, pred_sgdc, target_names=encoder.classes_))

# #### Comparing SGDClassifier to MultinomialNB, the accuracy of the SGDClassifier model was higher by 0.028, therefore I chosed the SGDClassifier to perform the next step which was hypertuning
# # Hypertuning
# using RandomSearch to find the best combination of parameters for building the model by randomly selecting a set of parameters


# # Create regularization penalty space
# penalty = ['l1', 'l2']

# # Create regularization max_iter
# max_iter = [5, 100, 1000] 

# # Create regularization max_iter
# alpha = [1e-3, 1e-4, 1e-5] 

# # Create regularization tol
# tol = [1e-3, None, 1e-5] 

# #
# loss=['hinge','log']

# # Create hyperparameter options
# hyperparameters = dict(alpha=alpha, penalty=penalty, max_iter=max_iter, tol=tol)
# mo_ran = RandomizedSearchCV(sgdc_, hyperparameters, random_state=1, n_iter=10, cv=5, verbose=0, n_jobs=-1)
# # Fit randomized search
# ran_fit = mo_ran.fit(x_word_count_vector, y_train)
# ran_fit.best_params_
# pred_ran = ran_fit.predict(x_testing_count)
# print('Accurary of SGDClass+RandomSearch (optimum parameter):', accuracy_score(y_test,pred_ran))
# print(classification_report(y_test, pred_ran, target_names=encoder.classes_))
sgdc_b=SGDClassifier(random_state=42,penalty= 'l2',max_iter= 1000,alpha= 0.0001)
sgdc_b.fit(x_word_count_vector, y_train)
pred_sgdc = sgdc_b.predict(x_testing_count)
# # With RandomSearch, we could improve accuracy of SGDClassifier by 0.004 and it's about 0.855

# ### Create data frame to store the predictions
y_new=pd.DataFrame(y_test)
inx=y_new.index.tolist()
for i in inx:
    y_new.loc[i,'article_topic']=data_new.loc[i,'article_topic']
# y_new['idx_pred']=pred_ran
y_new['idx_pred']=pred_sgdc
y_new['pred']=list(encoder.inverse_transform(pred_sgdc))
# print(y_new.head(20))


# ### Display top 10 words that occured most frequently in each topic's articles
# ## model with optimum parameters
# sgdc_b=SGDClassifier(random_state=42,penalty='l2', tol=None, max_iter=1000, alpha=0.0001)
# sgdc_b.fit(x_word_count_vector, y_train)

reverse_vo = {}
vocab = cv.vocabulary_
for word in vocab:
    index = vocab[word]
    reverse_vo[index] = word


coefs = sgdc_b.coef_
target_names = encoder.classes_
print('list of most frequent words :')
for i in range(len(target_names)):
    words = []
    for j in coefs[i].argsort()[-10:]:
        words.append(reverse_vo[j])
    print (target_names[i], '-', words, "\n")


# ### Predict articles' topics
# inp=data_new.article_content.loc[355]
##to try different input, choose one below, please notice if you want to activate the line, disable the line above
# inp={'article_content':['lucu sih teamsusahmoveon suara nih','rilis comeback mv jul jun','novanto bener periksa narapidana']}
# inp={'article_content':[data_new.article_new.loc[355],data_new.article_new.loc[356]]}
# inp=['lucu sih teamsusahmoveon suara nih','rilis comeback mv jul jun','novanto bener periksa narapidana']
# inp='lucu sih teamsusahmoveon suara nih'

# if isinstance(inp, pd.DataFrame) is False:                
#     if type(inp)!=dict:
#         if type(inp)==str:
#             inp=[inp]
#         if len(inp)!=1:
#             inp=pd.DataFrame(inp)
#         else:
#             inp=pd.Series(inp)
#     else:
#         if len(inp.values())!=1:
#             inp=pd.DataFrame(inp.values())
#         else:
#             inp=list(inp.values())
#             inp=inp[0]
#             inp=pd.DataFrame(inp)

# # Data selection for dataframe and series
# if isinstance(inp, pd.DataFrame):
#     artic=inp.iloc[0,0]
#     d_artic=inp.iloc[:,0]
# else:
#     artic=inp.iloc[0]
#     d_artic=inp.values
    
# # Stemming and stopword removing
# if artic not in data_new.article_new.values:
#     xy=[]
#     for i in d_artic:
#         stops_=stopword.remove(stemmer.stem(i))
#         wordy_=''
#         for st in stops_.split(' '):
#             if st.isalpha():      
#                 wordy_+=st+' '
#         xy.append(wordy_)
# else:
#     xy=inp

# x_t = cv.transform(xy)
# print('prediction:')
# print(list(encoder.inverse_transform(ran_fit.predict(x_t))))
# print('true topic:')
# data_new['article_topic'].loc[355]


# # - MODEL

# Models for text classification I used were MultinomialNB and SGDClassifier because MultinomialNB works well with discrete features such as word counts and SGDClassifier works well with data represented as dense. For this experiment, SGDClassifier with no paramater used got a higher accuracy score, therefore I  did hyperparameter tuning to get a better SGDClassifier model by trying a pair of parameters and finding the optimal one.

# # - VALIDATION

# I used accuracy because it indicated how good a model was to predict data correctly and I used classification report which also displayed f1 score, recall, prediction, and support (the actual number of occurrences of each class in data we predict) because it made us easier to understand the model perfomance and how good the model predicted some data dispersively. For example, in the classification report of MultinomialNB, it revealed that topic 'Horor' got 0 for f1 score, recall and prediction which meant that the model failed to predict all horor articles, whereas the number of horror articles (support) was 4.

# # - FUTURE RESEARCH

# Stemming in Bahasa Indonesia is hard, because the data of possible affix combinations and root forms of words is found limited and we may face some problems, such as word sense ambiguity, for example, the word 'berikan' can be chopped as 'ber-i-kan' ('i' will be stored as the root form of word) or 'beri-kan' ('beri' will be extracted) or 'ber-ikan' ('ikan' will be returned instead) or for this case, 'belasan' will be stored as 'bas'. Another example, we can't identify name such as 'Aqilah' (it will be truncated as 'Aqil-ah' and 'Aqil' will be extracted) or in this case, 'Mekkah' as 'Mek'. So for the next research, we suppose to do correction manually for words that are falsely stated from the stemming process. Manually here means creating a function where we would define some root forms of words that may be inaccessible in sastrawi library and manually describe the root forms of the words instead. We can also try boosting method to improve accuracy score 

# Saving model to disk
pickle.dump(sgdc_b, open('model_news.pkl','wb'))
pickle.dump(cv,open('news_cv.pkl','wb'))
pickle.dump(encoder,open('enc_news.pkl','wb'))
# # Loading model to compare the results
# model = pickle.load(open('model_news.pkl','rb'))
# cv_=pickle.load(open('news_cv.pkl','rb'))
# enc_=pickle.load(open('enc_news.pkl','rb'))
# y_for_test='lucu sih teamsusahmoveon suara nih','rilis comeback mv jul jun','novanto bener periksa narapidana'
# y_for_test=pd.Series(y_for_test)

# xy=[]
# for i in y_for_test.values:
#     stops_=stopword.remove(stemmer.stem(i))
#     wordy_=''
#     for st in stops_.split(' '):
#         if st.isalpha():      
#             wordy_+=st+' '
#     xy.append(wordy_)
# x_t=cv_.transform(xy)
# resu=model.predict(x_t)
# print('prediction:')
# s = [str(i) for i in list(enc_.inverse_transform(resu))] 
# res = ", ".join(s)   
# print(res)
  