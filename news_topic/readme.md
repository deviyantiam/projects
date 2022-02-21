
The project is to identify topic of news


```python
# !pip install PySastrawi
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer#, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV
```

# Data Preparation
prepare the dataset by gathering and checking if theres any missing value and remove them
## Data Gathering


```python
## collecting data
data_csv=pd.read_csv('data.csv')
print('Shape of dataset =', data_csv.shape)
data_csv.head(5)
```

    Shape of dataset = (10000, 3)





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
      <th>article_id</th>
      <th>article_topic</th>
      <th>article_content</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>93205794</td>
      <td>Internasional</td>
      <td>Kepolisian Inggris tengah memburu pelaku yang...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>93186698</td>
      <td>Ekonomi</td>
      <td>Seluruh layanan transaksi di jalan tol akan m...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>93191463</td>
      <td>Teknologi</td>
      <td>\nHari ini, Rabu (23/8), ternyata menjadi har...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>93219292</td>
      <td>Ekonomi</td>
      <td>Saat ini Indonesia hanya memiliki cadangan ba...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>343106</td>
      <td>Hiburan</td>
      <td>Hari ini, Selasa (1/8), pedangdut Ridho Rhoma...</td>
    </tr>
  </tbody>
</table>
</div>



## Data Cleaning


```python
## check missing value
data_csv.isnull().sum()
```




    article_id          0
    article_topic       0
    article_content    36
    dtype: int64




```python
## check any cells that have missing value
data_csv[data_csv['article_content'].isna()]
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
      <th>article_id</th>
      <th>article_topic</th>
      <th>article_content</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>197</th>
      <td>93210288</td>
      <td>Teknologi</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>674</th>
      <td>93185319</td>
      <td>Hiburan</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>817</th>
      <td>93189481</td>
      <td>Hiburan</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>972</th>
      <td>93184085</td>
      <td>Otomotif</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2015</th>
      <td>93195291</td>
      <td>Hiburan</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2250</th>
      <td>93201544</td>
      <td>Hiburan</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3276</th>
      <td>93213891</td>
      <td>Teknologi</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4150</th>
      <td>93197166</td>
      <td>Hiburan</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4338</th>
      <td>93186717</td>
      <td>Sepak Bola</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4750</th>
      <td>93224988</td>
      <td>Sepak Bola</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4838</th>
      <td>93183169</td>
      <td>Politik</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4917</th>
      <td>93194456</td>
      <td>Hiburan</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5126</th>
      <td>93184755</td>
      <td>Hiburan</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5306</th>
      <td>93195757</td>
      <td>Lifestyle</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5859</th>
      <td>93208751</td>
      <td>Hiburan</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5965</th>
      <td>93189109</td>
      <td>Lifestyle</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6032</th>
      <td>93201441</td>
      <td>Internasional</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6094</th>
      <td>93187589</td>
      <td>Hiburan</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6442</th>
      <td>947838</td>
      <td>Ekonomi</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6539</th>
      <td>93181264</td>
      <td>Lifestyle</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6689</th>
      <td>845044</td>
      <td>Hiburan</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6760</th>
      <td>93188937</td>
      <td>Hiburan</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6816</th>
      <td>93195181</td>
      <td>Internasional</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6886</th>
      <td>93201878</td>
      <td>Sepak Bola</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7239</th>
      <td>93187815</td>
      <td>Hiburan</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7307</th>
      <td>93182251</td>
      <td>Hiburan</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7403</th>
      <td>93186538</td>
      <td>Hiburan</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7425</th>
      <td>93186516</td>
      <td>Sepak Bola</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7490</th>
      <td>93192172</td>
      <td>Teknologi</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7526</th>
      <td>93203404</td>
      <td>Hiburan</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7920</th>
      <td>93208905</td>
      <td>Hiburan</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8457</th>
      <td>93201720</td>
      <td>Teknologi</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8833</th>
      <td>93189224</td>
      <td>Ekonomi</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8851</th>
      <td>93191464</td>
      <td>Lifestyle</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9552</th>
      <td>93183963</td>
      <td>Hiburan</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9696</th>
      <td>93201912</td>
      <td>Hiburan</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
## drop those cells
data_csv=data_csv.dropna(how='any')
data_csv.isna().sum()
```




    article_id         0
    article_topic      0
    article_content    0
    dtype: int64




```python
## find unique values from 'article_topic'
topi=data_csv['article_topic'].unique().tolist()
print('List of topics:')
print(*topi, sep=', ')
```

    List of topics:
    Internasional, Ekonomi, Teknologi, Hiburan, Haji, Travel, Personal, Sepak Bola, Health, Sports, Politik, Otomotif, KPK, Lifestyle, Keuangan, Sejarah, Regional, Pendidikan, Hukum, Obat-obatan, Bojonegoro, Kesehatan, Horor, Bisnis, MotoGP, Sains, Jakarta, Pilgub Jatim, K-Pop



```python
## class distribution
for p in list(set(data_csv.article_topic)):
    print('number of',p,' ',len(data_csv.loc[data_csv['article_topic'] == p]))
```

    number of Pilgub Jatim   25
    number of Hukum   85
    number of Horor   50
    number of Sports   435
    number of Otomotif   173
    number of Keuangan   14
    number of Politik   103
    number of MotoGP   35
    number of Sains   174
    number of Jakarta   12
    number of Health   131
    number of Sejarah   70
    number of Haji   1497
    number of Sepak Bola   1180
    number of Internasional   739
    number of Bojonegoro   260
    number of Teknologi   567
    number of Regional   35
    number of K-Pop   61
    number of Bisnis   25
    number of Kesehatan   195
    number of Personal   81
    number of Hiburan   1448
    number of KPK   37
    number of Ekonomi   1760
    number of Lifestyle   568
    number of Travel   76
    number of Obat-obatan   58
    number of Pendidikan   70



```python
## convert categorical values into numeric values
encoder = LabelEncoder()
encoder.fit(data_csv['article_topic'])
data_csv['index_topic']= encoder.transform(data_csv['article_topic'])
data_csv.head()
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
      <th>article_id</th>
      <th>article_topic</th>
      <th>article_content</th>
      <th>index_topic</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>93205794</td>
      <td>Internasional</td>
      <td>Kepolisian Inggris tengah memburu pelaku yang...</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>93186698</td>
      <td>Ekonomi</td>
      <td>Seluruh layanan transaksi di jalan tol akan m...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>93191463</td>
      <td>Teknologi</td>
      <td>\nHari ini, Rabu (23/8), ternyata menjadi har...</td>
      <td>27</td>
    </tr>
    <tr>
      <th>3</th>
      <td>93219292</td>
      <td>Ekonomi</td>
      <td>Saat ini Indonesia hanya memiliki cadangan ba...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>343106</td>
      <td>Hiburan</td>
      <td>Hari ini, Selasa (1/8), pedangdut Ridho Rhoma...</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



# Feature Engineering

raw text of training and test data will be transformed into new feature by going through some processes below. Stopword removing is to remove the most common words in a language and stemming is to reduce a word by remove its affixes. After re-runing some process multiple times, we noticed there were rows that had the same article contents. Therefore, the latest version of the script for stemming and stopword removing was created as follows:


```python
## import StemmerFactory class
factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()

## create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

## create new column of stemmed and stopwords removed articles
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
```

#### After multiple process of trial and error, I found that there were a few duplicate articles, eg below shows that row 791 and 107 were the same. That's why i put this command in cell above
``` pyhton
    indx=data_csv.loc[data_csv['article_content']==i].index.tolist()
    data_csv.loc[indx,'article_new']=wordy
```


```python
data_csv.loc[data_csv['article_content']==data_csv['article_content'][791]]
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
      <th>article_id</th>
      <th>article_topic</th>
      <th>article_content</th>
      <th>index_topic</th>
      <th>article_new</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>107</th>
      <td>1599799</td>
      <td>Haji</td>
      <td>KBRN, Madiun (MCH) : Kepala kantor Kementeria...</td>
      <td>3</td>
      <td>kbrn madiun mch kepala kantor menteri agama ke...</td>
    </tr>
    <tr>
      <th>791</th>
      <td>93181816</td>
      <td>Haji</td>
      <td>KBRN, Madiun (MCH) : Kepala kantor Kementeria...</td>
      <td>3</td>
      <td>kbrn madiun mch kepala kantor menteri agama ke...</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_csv.loc[data_csv['article_content']==data_csv['article_content'][791]].index[0]
```




    107




```python
data_csv.loc[data_csv['article_content']==data_csv['article_content'][791]].index[1]
```




    791




```python
## splitting
X_train, X_test, y_train, y_test = train_test_split(data_csv['article_new'], data_csv['index_topic'], test_size=.15, random_state = 79)
print("Training dataset: ", X_train.shape[0])
print("Test dataset: ", X_test.shape[0])
```

    Training dataset:  8469
    Test dataset:  1495


```python
##  instantiate CountVectorizer()
cv = CountVectorizer()
 
## this steps generates word counts for the words in your dataset
x_word_count_vector = cv.fit_transform(X_train)
```
#### After splitting, we ran the cell above and we got an error message: 
```python
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-13-6036975bf677> in <module>
      3 
      4 ## this steps generates word counts for the words in your dataset
----> 5 x_word_count_vector = cv.fit_transform(X_train)
      6 ## word_count_vector

~/anaconda3/lib/python3.7/site-packages/sklearn/feature_extraction/text.py in fit_transform(self, raw_documents, y)
   1029 
   1030         vocabulary, X = self._count_vocab(raw_documents,
-> 1031                                           self.fixed_vocabulary_)
   1032 
   1033         if self.binary:

~/anaconda3/lib/python3.7/site-packages/sklearn/feature_extraction/text.py in _count_vocab(self, raw_documents, fixed_vocab)
    941         for doc in raw_documents:
    942             feature_counter = {}
--> 943             for feature in analyze(doc):
    944                 try:
    945                     feature_idx = vocabulary[feature]

~/anaconda3/lib/python3.7/site-packages/sklearn/feature_extraction/text.py in <lambda>(doc)
    327                                                tokenize)
    328             return lambda doc: self._word_ngrams(
--> 329                 tokenize(preprocess(self.decode(doc))), stop_words)
    330 
    331         else:

~/anaconda3/lib/python3.7/site-packages/sklearn/feature_extraction/text.py in <lambda>(x)
    255 
    256         if self.lowercase:
--> 257             return lambda x: strip_accents(x.lower())
    258         else:
    259             return strip_accents

AttributeError: 'float' object has no attribute 'lower'
```
#### it turned out there were missing values in some rows, we would analize how it could be possible. We already removed the missing value though. Perhaps, we would track down where the problem was


```python
## check if there's any string cell contains blank space
id_=data_csv[data_csv['article_new']==''].index.tolist()
data_csv[data_csv['article_new']=='']
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
      <th>article_id</th>
      <th>article_topic</th>
      <th>article_content</th>
      <th>index_topic</th>
      <th>article_new</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>555</th>
      <td>93181418</td>
      <td>Hiburan</td>
      <td></td>
      <td>5</td>
      <td></td>
    </tr>
    <tr>
      <th>1331</th>
      <td>93191657</td>
      <td>Hiburan</td>
      <td>,</td>
      <td>5</td>
      <td></td>
    </tr>
    <tr>
      <th>1432</th>
      <td>93212920</td>
      <td>Kesehatan</td>
      <td></td>
      <td>12</td>
      <td></td>
    </tr>
    <tr>
      <th>1601</th>
      <td>93181411</td>
      <td>Hiburan</td>
      <td></td>
      <td>5</td>
      <td></td>
    </tr>
    <tr>
      <th>2031</th>
      <td>93210296</td>
      <td>Health</td>
      <td>.,</td>
      <td>4</td>
      <td></td>
    </tr>
    <tr>
      <th>2342</th>
      <td>93190978</td>
      <td>Hiburan</td>
      <td></td>
      <td>5</td>
      <td></td>
    </tr>
    <tr>
      <th>2937</th>
      <td>93190761</td>
      <td>Hiburan</td>
      <td>.</td>
      <td>5</td>
      <td></td>
    </tr>
    <tr>
      <th>3600</th>
      <td>1485824</td>
      <td>Hiburan</td>
      <td></td>
      <td>5</td>
      <td></td>
    </tr>
    <tr>
      <th>3730</th>
      <td>93191661</td>
      <td>Hiburan</td>
      <td>,</td>
      <td>5</td>
      <td></td>
    </tr>
    <tr>
      <th>4085</th>
      <td>1586004</td>
      <td>Politik</td>
      <td></td>
      <td>21</td>
      <td></td>
    </tr>
    <tr>
      <th>4087</th>
      <td>4052810</td>
      <td>Politik</td>
      <td></td>
      <td>21</td>
      <td></td>
    </tr>
    <tr>
      <th>4173</th>
      <td>93191699</td>
      <td>Hiburan</td>
      <td>.</td>
      <td>5</td>
      <td></td>
    </tr>
    <tr>
      <th>5297</th>
      <td>93216759</td>
      <td>Kesehatan</td>
      <td></td>
      <td>12</td>
      <td></td>
    </tr>
    <tr>
      <th>6352</th>
      <td>1484488</td>
      <td>Sepak Bola</td>
      <td></td>
      <td>25</td>
      <td></td>
    </tr>
    <tr>
      <th>6887</th>
      <td>93181415</td>
      <td>Hiburan</td>
      <td></td>
      <td>5</td>
      <td></td>
    </tr>
    <tr>
      <th>6942</th>
      <td>93188749</td>
      <td>Politik</td>
      <td></td>
      <td>21</td>
      <td></td>
    </tr>
    <tr>
      <th>8074</th>
      <td>93190979</td>
      <td>Hiburan</td>
      <td></td>
      <td>5</td>
      <td></td>
    </tr>
    <tr>
      <th>8697</th>
      <td>93195225</td>
      <td>Kesehatan</td>
      <td></td>
      <td>12</td>
      <td></td>
    </tr>
    <tr>
      <th>9261</th>
      <td>93191674</td>
      <td>Hiburan</td>
      <td>.</td>
      <td>5</td>
      <td></td>
    </tr>
    <tr>
      <th>9899</th>
      <td>93228169</td>
      <td>Personal</td>
      <td>Hello</td>
      <td>19</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>




```python
## replace blank space with np.nan
for i in id_:
    data_csv.loc[i] = data_csv.loc[i].replace('',np.nan)
```


```python
data_csv[data_csv['article_new'].isnull()]
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
      <th>article_id</th>
      <th>article_topic</th>
      <th>article_content</th>
      <th>index_topic</th>
      <th>article_new</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>555</th>
      <td>93181418</td>
      <td>Hiburan</td>
      <td></td>
      <td>5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1331</th>
      <td>93191657</td>
      <td>Hiburan</td>
      <td>,</td>
      <td>5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1432</th>
      <td>93212920</td>
      <td>Kesehatan</td>
      <td></td>
      <td>12</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1601</th>
      <td>93181411</td>
      <td>Hiburan</td>
      <td></td>
      <td>5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2031</th>
      <td>93210296</td>
      <td>Health</td>
      <td>.,</td>
      <td>4</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2342</th>
      <td>93190978</td>
      <td>Hiburan</td>
      <td></td>
      <td>5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2937</th>
      <td>93190761</td>
      <td>Hiburan</td>
      <td>.</td>
      <td>5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3600</th>
      <td>1485824</td>
      <td>Hiburan</td>
      <td></td>
      <td>5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3730</th>
      <td>93191661</td>
      <td>Hiburan</td>
      <td>,</td>
      <td>5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4085</th>
      <td>1586004</td>
      <td>Politik</td>
      <td></td>
      <td>21</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4087</th>
      <td>4052810</td>
      <td>Politik</td>
      <td></td>
      <td>21</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4173</th>
      <td>93191699</td>
      <td>Hiburan</td>
      <td>.</td>
      <td>5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5297</th>
      <td>93216759</td>
      <td>Kesehatan</td>
      <td></td>
      <td>12</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6352</th>
      <td>1484488</td>
      <td>Sepak Bola</td>
      <td></td>
      <td>25</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6887</th>
      <td>93181415</td>
      <td>Hiburan</td>
      <td></td>
      <td>5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6942</th>
      <td>93188749</td>
      <td>Politik</td>
      <td></td>
      <td>21</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8074</th>
      <td>93190979</td>
      <td>Hiburan</td>
      <td></td>
      <td>5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8697</th>
      <td>93195225</td>
      <td>Kesehatan</td>
      <td></td>
      <td>12</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9261</th>
      <td>93191674</td>
      <td>Hiburan</td>
      <td>.</td>
      <td>5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9899</th>
      <td>93228169</td>
      <td>Personal</td>
      <td>Hello</td>
      <td>19</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_csv.isnull().sum()
```




    article_id          0
    article_topic       0
    article_content     0
    index_topic         0
    article_new        20
    dtype: int64



#### There were rows whose 'article_content' only contained spaces '  '. 


```python
data_csv.article_content[9261]
```




    '  .'




```python
## drop those cells
data_new=data_csv.dropna(how='any')
data_new.isna().sum()
```




    article_id         0
    article_topic      0
    article_content    0
    index_topic        0
    article_new        0
    dtype: int64



### Split data to training and test data


```python
X_train, X_test, y_train, y_test = train_test_split(data_new['article_new'], data_new['index_topic'], test_size=.15, random_state = 89)
print("Training dataset: ", X_train.shape[0])
print("Test dataset: ", X_test.shape[0])
```

    Training dataset:  8452
    Test dataset:  1492



```python
##  instantiate CountVectorizer()
cv = CountVectorizer()
 
## this steps generates word counts for the words in your dataset
x_word_count_vector = cv.fit_transform(X_train)
```


```python
x_testing_count = cv.transform(X_test)
```

# Model Building


```python
na_bayes = MultinomialNB()
na_bayes.fit(x_word_count_vector, y_train)
```




    MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)




```python
pred_nb = na_bayes.predict(x_testing_count)
```


```python
sgdc_=SGDClassifier(random_state=42)
sgdc_.fit(x_word_count_vector, y_train)
```

    /Users/devi/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDClassifier in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)





    SGDClassifier(alpha=0.0001, average=False, class_weight=None,
           early_stopping=False, epsilon=0.1, eta0=0.0, fit_intercept=True,
           l1_ratio=0.15, learning_rate='optimal', loss='hinge', max_iter=None,
           n_iter=None, n_iter_no_change=5, n_jobs=None, penalty='l2',
           power_t=0.5, random_state=42, shuffle=True, tol=None,
           validation_fraction=0.1, verbose=0, warm_start=False)




```python
pred_sgdc = sgdc_.predict(x_testing_count)
```

## EVALUATION


```python
print('Accurary of Naive Bayes:', accuracy_score(y_test,pred_nb))
print(classification_report(y_test, pred_nb, target_names=encoder.classes_))
```

    Accurary of Naive Bayes: 0.8230563002680965
                   precision    recall  f1-score   support
    
           Bisnis       0.00      0.00      0.00         2
       Bojonegoro       0.78      0.83      0.81        30
          Ekonomi       0.87      0.97      0.92       280
             Haji       0.98      0.98      0.98       253
           Health       0.54      0.58      0.56        26
          Hiburan       0.79      0.99      0.88       188
            Horor       0.00      0.00      0.00         5
            Hukum       0.67      0.50      0.57         8
    Internasional       0.78      0.87      0.82       107
          Jakarta       0.00      0.00      0.00         3
            K-Pop       1.00      0.22      0.36         9
              KPK       1.00      1.00      1.00         3
        Kesehatan       0.50      0.61      0.55        33
         Keuangan       0.00      0.00      0.00         3
        Lifestyle       0.67      0.78      0.72        92
           MotoGP       0.00      0.00      0.00        10
      Obat-obatan       0.00      0.00      0.00        13
         Otomotif       1.00      0.67      0.80        18
       Pendidikan       1.00      0.31      0.47        13
         Personal       1.00      0.12      0.22         8
     Pilgub Jatim       0.00      0.00      0.00         5
          Politik       0.44      0.29      0.35        14
         Regional       0.00      0.00      0.00         2
            Sains       0.00      0.00      0.00        27
          Sejarah       0.00      0.00      0.00         8
       Sepak Bola       0.83      0.99      0.90       172
           Sports       0.62      0.42      0.50        62
        Teknologi       0.90      0.78      0.84        88
           Travel       0.00      0.00      0.00        10
    
        micro avg       0.82      0.82      0.82      1492
        macro avg       0.50      0.41      0.42      1492
     weighted avg       0.78      0.82      0.79      1492
    


    /Users/devi/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)



```python
print('Accurary of SGDC:', accuracy_score(y_test,pred_sgdc))
print(classification_report(y_test, pred_sgdc, target_names=encoder.classes_))
```

    Accurary of SGDC: 0.8512064343163539
                   precision    recall  f1-score   support
    
           Bisnis       0.50      1.00      0.67         2
       Bojonegoro       0.85      0.93      0.89        30
          Ekonomi       0.95      0.94      0.95       280
             Haji       0.97      1.00      0.98       253
           Health       0.38      0.35      0.36        26
          Hiburan       0.91      0.97      0.94       188
            Horor       1.00      0.60      0.75         5
            Hukum       0.55      0.75      0.63         8
    Internasional       0.82      0.82      0.82       107
          Jakarta       0.00      0.00      0.00         3
            K-Pop       0.56      0.56      0.56         9
              KPK       1.00      1.00      1.00         3
        Kesehatan       0.40      0.52      0.45        33
         Keuangan       1.00      0.67      0.80         3
        Lifestyle       0.84      0.70      0.76        92
           MotoGP       0.73      0.80      0.76        10
      Obat-obatan       0.38      0.38      0.38        13
         Otomotif       1.00      0.83      0.91        18
       Pendidikan       0.73      0.62      0.67        13
         Personal       0.43      0.38      0.40         8
     Pilgub Jatim       0.31      0.80      0.44         5
          Politik       1.00      0.36      0.53        14
         Regional       0.00      0.00      0.00         2
            Sains       0.76      0.81      0.79        27
          Sejarah       0.33      0.12      0.18         8
       Sepak Bola       0.82      0.96      0.89       172
           Sports       0.74      0.42      0.54        62
        Teknologi       0.95      0.86      0.90        88
           Travel       0.67      0.60      0.63        10
    
        micro avg       0.85      0.85      0.85      1492
        macro avg       0.67      0.65      0.64      1492
     weighted avg       0.86      0.85      0.85      1492
    


#### Comparing SGDClassifier to MultinomialNB, the accuracy of the SGDClassifier model was higher by 0.028, therefore I chosed the SGDClassifier to perform the next step which was hypertuning
# Hypertuning
using RandomSearch to find the best combination of parameters for building the model by randomly selecting a set of parameters


```python
# Create regularization penalty space
penalty = ['l1', 'l2']

# Create regularization max_iter
max_iter = [5, 100, 1000] 

# Create regularization max_iter
alpha = [1e-3, 1e-4, 1e-5] 

# Create regularization tol
tol = [1e-3, None, 1e-5] 

#
loss=['hinge','log']

# Create hyperparameter options
hyperparameters = dict(alpha=alpha, penalty=penalty, max_iter=max_iter, tol=tol)
```


```python
mo_ran = RandomizedSearchCV(sgdc_, hyperparameters, random_state=1, n_iter=10, cv=5, verbose=0, n_jobs=-1)
# Fit randomized search
ran_fit = mo_ran.fit(x_word_count_vector, y_train)
```

    /Users/devi/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/stochastic_gradient.py:183: FutureWarning: max_iter and tol parameters have been added in SGDClassifier in 0.19. If max_iter is set but tol is left unset, the default value for tol in 0.19 and 0.20 will be None (which is equivalent to -infinity, so it has no effect) but will change in 0.21 to 1e-3. Specify tol to silence this warning.
      FutureWarning)



```python
ran_fit.best_params_
```




    {'tol': None, 'penalty': 'l2', 'max_iter': 1000, 'alpha': 0.0001}




```python
pred_ran = ran_fit.predict(x_testing_count)
```


```python
print('Accurary of SGDClass+RandomSearch (optimum parameter):', accuracy_score(y_test,pred_ran))
```

    Accurary of SGDClass+RandomSearch (optimum parameter): 0.8552278820375335



```python
print(classification_report(y_test, pred_ran, target_names=encoder.classes_))
```

                   precision    recall  f1-score   support
    
           Bisnis       0.33      0.50      0.40         2
       Bojonegoro       0.93      0.90      0.92        30
          Ekonomi       0.93      0.97      0.95       280
             Haji       0.98      0.99      0.99       253
           Health       0.29      0.31      0.30        26
          Hiburan       0.93      0.96      0.95       188
            Horor       0.80      0.80      0.80         5
            Hukum       0.46      0.75      0.57         8
    Internasional       0.89      0.86      0.88       107
          Jakarta       0.00      0.00      0.00         3
            K-Pop       0.73      0.89      0.80         9
              KPK       0.75      1.00      0.86         3
        Kesehatan       0.39      0.45      0.42        33
         Keuangan       1.00      0.33      0.50         3
        Lifestyle       0.83      0.76      0.80        92
           MotoGP       1.00      0.70      0.82        10
      Obat-obatan       0.33      0.15      0.21        13
         Otomotif       0.82      0.78      0.80        18
       Pendidikan       0.88      0.54      0.67        13
         Personal       0.38      0.38      0.38         8
     Pilgub Jatim       0.57      0.80      0.67         5
          Politik       0.54      0.50      0.52        14
         Regional       0.00      0.00      0.00         2
            Sains       0.76      0.96      0.85        27
          Sejarah       0.60      0.38      0.46         8
       Sepak Bola       0.82      0.90      0.86       172
           Sports       0.64      0.48      0.55        62
        Teknologi       0.94      0.85      0.89        88
           Travel       0.75      0.60      0.67        10
    
        micro avg       0.86      0.86      0.86      1492
        macro avg       0.66      0.64      0.64      1492
     weighted avg       0.85      0.86      0.85      1492
    


    /Users/devi/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)


With RandomSearch, we could improve accuracy of SGDClassifier by 0.004 and it's about 0.855
### Create data frame to store the predictions


```python
y_new=pd.DataFrame(y_test)
```


```python
inx=y_new.index.tolist()
```


```python
for i in inx:
    y_new.loc[i,'article_topic']=data_new.loc[i,'article_topic']
```


```python
y_new['idx_pred']=pred_ran
```


```python
y_new['pred']=list(encoder.inverse_transform(pred_ran))
```


```python
y_new.head(20)
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
      <th>index_topic</th>
      <th>article_topic</th>
      <th>idx_pred</th>
      <th>pred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6472</th>
      <td>3</td>
      <td>Haji</td>
      <td>3</td>
      <td>Haji</td>
    </tr>
    <tr>
      <th>5277</th>
      <td>1</td>
      <td>Bojonegoro</td>
      <td>0</td>
      <td>Bisnis</td>
    </tr>
    <tr>
      <th>601</th>
      <td>3</td>
      <td>Haji</td>
      <td>3</td>
      <td>Haji</td>
    </tr>
    <tr>
      <th>5130</th>
      <td>25</td>
      <td>Sepak Bola</td>
      <td>25</td>
      <td>Sepak Bola</td>
    </tr>
    <tr>
      <th>9220</th>
      <td>3</td>
      <td>Haji</td>
      <td>3</td>
      <td>Haji</td>
    </tr>
    <tr>
      <th>6305</th>
      <td>5</td>
      <td>Hiburan</td>
      <td>5</td>
      <td>Hiburan</td>
    </tr>
    <tr>
      <th>9157</th>
      <td>26</td>
      <td>Sports</td>
      <td>26</td>
      <td>Sports</td>
    </tr>
    <tr>
      <th>872</th>
      <td>2</td>
      <td>Ekonomi</td>
      <td>2</td>
      <td>Ekonomi</td>
    </tr>
    <tr>
      <th>2779</th>
      <td>2</td>
      <td>Ekonomi</td>
      <td>2</td>
      <td>Ekonomi</td>
    </tr>
    <tr>
      <th>7135</th>
      <td>25</td>
      <td>Sepak Bola</td>
      <td>25</td>
      <td>Sepak Bola</td>
    </tr>
    <tr>
      <th>3969</th>
      <td>12</td>
      <td>Kesehatan</td>
      <td>12</td>
      <td>Kesehatan</td>
    </tr>
    <tr>
      <th>5476</th>
      <td>3</td>
      <td>Haji</td>
      <td>3</td>
      <td>Haji</td>
    </tr>
    <tr>
      <th>9850</th>
      <td>25</td>
      <td>Sepak Bola</td>
      <td>25</td>
      <td>Sepak Bola</td>
    </tr>
    <tr>
      <th>95</th>
      <td>4</td>
      <td>Health</td>
      <td>12</td>
      <td>Kesehatan</td>
    </tr>
    <tr>
      <th>955</th>
      <td>20</td>
      <td>Pilgub Jatim</td>
      <td>21</td>
      <td>Politik</td>
    </tr>
    <tr>
      <th>2177</th>
      <td>25</td>
      <td>Sepak Bola</td>
      <td>25</td>
      <td>Sepak Bola</td>
    </tr>
    <tr>
      <th>8491</th>
      <td>2</td>
      <td>Ekonomi</td>
      <td>2</td>
      <td>Ekonomi</td>
    </tr>
    <tr>
      <th>3414</th>
      <td>18</td>
      <td>Pendidikan</td>
      <td>23</td>
      <td>Sains</td>
    </tr>
    <tr>
      <th>7672</th>
      <td>2</td>
      <td>Ekonomi</td>
      <td>2</td>
      <td>Ekonomi</td>
    </tr>
    <tr>
      <th>1661</th>
      <td>3</td>
      <td>Haji</td>
      <td>3</td>
      <td>Haji</td>
    </tr>
  </tbody>
</table>
</div>



### Display top 10 words that occured most frequently in each topic's articles


```python
## model with optimum parameters
sgdc_b=SGDClassifier(random_state=42,penalty='l2', tol=None, max_iter=1000, alpha=0.0001)
sgdc_b.fit(x_word_count_vector, y_train)
```




    SGDClassifier(alpha=0.0001, average=False, class_weight=None,
           early_stopping=False, epsilon=0.1, eta0=0.0, fit_intercept=True,
           l1_ratio=0.15, learning_rate='optimal', loss='hinge', max_iter=1000,
           n_iter=None, n_iter_no_change=5, n_jobs=None, penalty='l2',
           power_t=0.5, random_state=42, shuffle=True, tol=None,
           validation_fraction=0.1, verbose=0, warm_start=False)




```python
reverse_vo = {}
vocab = cv.vocabulary_
for word in vocab:
    index = vocab[word]
    reverse_vo[index] = word
```


```python
coefs = sgdc_b.coef_
target_names = encoder.classes_
print('list of most frequent words :')
for i in range(len(target_names)):
    words = []
    for j in coefs[i].argsort()[-10:]:
        words.append(reverse_vo[j])
    print (target_names[i], '-', words, "\n")
```

    list of most frequent words :
    Bisnis - ['steak', 'beliau', 'rintis', 'tapcash', 'slack', 'bisnis', 'lucy', 'buka', 'era', 'sapi'] 
    
    Bojonegoro - ['lis', 'putar', 'berita', 'mu', 'html', 'link', 'com', 'reporter', 'blokbojonegoro', 'bojonegoro'] 
    
    Ekonomi - ['kumpar', 'achsien', 'bendung', 'kutip', 'yg', 'persero', 'bank', 'esdm', 'selasa', 'menteri'] 
    
    Haji - ['kemenag', 'jamaah', 'id', 'co', 'https', 'read', 'sumber', 'haji', 'mch', 'tes'] 
    
    Health - ['haid', 'herbal', 'efek', 'koreng', 'ubah', 'nasofaring', 'risiko', 'nyamuk', 'nyaman', 'alas'] 
    
    Hiburan - ['ratnacece', 'suara', 'nih', 'kalo', 'nyanyi', 'teamsusahmoveon', 'hahahaha', 'team', 'sih', 'lucu'] 
    
    Horor - ['ekor', 'mistis', 'lendra', 'wujud', 'jelma', 'aktifitas', 'frasa', 'tasyakkul', 'jinn', 'dimensi'] 
    
    Hukum - ['mahkamah', 'febri', 'daring', 'korupsi', 'sangka', 'narapidana', 'novel', 'periksa', 'novanto', 'bener'] 
    
    Internasional - ['kamis', 'rabu', 'kiamat', 'tewas', 'polisi', 'lansir', 'press', 'reuters', 'associated', 'kutip'] 
    
    Jakarta - ['publik', 'run', 'but', 'jakarta', 'semper', 'that', 'limbah', 'you', 'peta', 'lurah'] 
    
    K-Pop - ['jung', 'chingudeul', 'debut', 'group', 'tanggal', 'anggota', 'ulang', 'mv', 'rilis', 'comeback'] 
    
    KPK - ['ta', 'tanda', 'juara', 'duduk', 'lumpur', 'malang', 'koruptor', 'topeng', 'tpk', 'segel'] 
    
    Kesehatan - ['dok', 'jelly', 'bengkak', 'sendi', 'organ', 'radang', 'lutut', 'tuntas', 'efeksamping', 'tubuhm'] 
    
    Keuangan - ['uang', 'bijak', 'ilegal', 'sistem', 'management', 'manfaat', 'milik', 'investasi', 'pemprov', 'dki'] 
    
    Lifestyle - ['cantik', 'us', 'kumpar', 'dr', 'percaya', 'pikir', 'santap', 'makeup', 'sejati', 'lansir'] 
    
    MotoGP - ['cepat', 'lambat', 'petrucci', 'motogp', 'motor', 'misano', 'pedrosa', 'vinales', 'sirkuit', 'crash'] 
    
    Obat-obatan - ['miom', 'kandung', 'formulasi', 'wasir', 'anus', 'komedo', 'hati', 'batuk', 'endometrium', 'tubuh'] 
    
    Otomotif - ['cewek', 'lomba', 'sari', 'cowok', 'debu', 'halang', 'terik', 'pekat', 'gue', 'gaya'] 
    
    Pendidikan - ['cabut', 'bincang', 'sunat', 'hehe', 'takut', 'hoax', 'nisan', 'placophobia', 'hapus', 'sayang'] 
    
    Personal - ['kitamenjadi', 'iramameniti', 'kamusaat', 'tapak', 'lupa', 'suara', 'bait', 'tuh', 'hehehhe', 'hellojikaahshhshdhdjjdshhdjdjjdjdjjd'] 
    
    Pilgub Jatim - ['rendra', 'calon', 'bangsaonline', 'gubernur', 'timur', 'pdip', 'bu', 'khofifah', 'risma', 'jatim'] 
    
    Politik - ['ubk', 'pilkada', 'daerah', 'politik', 'probolinggo', 'tolong', 'tau', 'sholat', 'inspiring', 'quotes'] 
    
    Regional - ['ansor', 'gresik', 'pemkot', 'trilyun', 'probolinggo', 'dinas', 'mojokerto', 'bangsaonline', 'pgri', 'pacitan'] 
    
    Sains - ['kode', 'tutup', 'distraksi', 'serangga', 'indra', 'cahaya', 'robot', 'umami', 'sumber', 'gambar'] 
    
    Sejarah - ['pandu', 'oktoberfest', 'gera', 'moguel', 'nama', 'tan', 'kolonial', 'blambangan', 'belanda', 'sejarah'] 
    
    Sepak Bola - ['adoring', 'fandom', 'mimpi', 'olahraga', 'konyol', 'selamat', 'informasi', 'pusat', 'test', 'sporttoday'] 
    
    Sports - ['duka', 'matchday', 'carvajal', 'swedia', 'allegri', 'vettel', 'detik', 'atlet', 'chamberlain', 'gp'] 
    
    Teknologi - ['jack', 'warganet', 'milik', 'situs', 'twitter', 'karakter', 'fitur', 'tweet', 'hak', 'cipta'] 
    
    Travel - ['libur', 'gaya', 'banyuwangi', 'festival', 'indah', 'lombok', 'getaway', 'traveling', 'belanak', 'selong'] 
    


### Predict articles' topics


```python
# inp=data_new.article_content.loc[355]
##to try different input, choose one below, please notice if you want to activate the line, disable the line above
# inp={'article_content':['lucu sih teamsusahmoveon suara nih','rilis comeback mv jul jun','novanto bener periksa narapidana']}
# inp={'article_content':[data_new.article_new.loc[355],data_new.article_new.loc[356]]}
inp=['lucu sih teamsusahmoveon suara nih','rilis comeback mv jul jun','novanto bener periksa narapidana']
# inp='lucu sih teamsusahmoveon suara nih'


if isinstance(inp, pd.DataFrame) is False:                
    if type(inp)!=dict:
        if type(inp)==str:
            inp=[inp]
        if len(inp)!=1:
            inp=pd.DataFrame(inp)
        else:
            inp=pd.Series(inp)
    else:
        if len(inp.values())!=1:
            inp=pd.DataFrame(inp.values())
        else:
            inp=list(inp.values())
            inp=inp[0]
            inp=pd.DataFrame(inp)

# Data selection for dataframe and series
if isinstance(inp, pd.DataFrame):
    artic=inp.iloc[0,0]
    d_artic=inp.iloc[:,0]
else:
    artic=inp.iloc[0]
    d_artic=inp.values
    
# Stemming and stopword removing
if artic not in data_new.article_new.values:
    xy=[]
    for i in d_artic:
        stops_=stopword.remove(stemmer.stem(i))
        wordy_=''
        for st in stops_.split(' '):
            if st.isalpha():      
                wordy_+=st+' '
        xy.append(wordy_)
else:
    xy=inp
```


```python
x_t = cv.transform(xy)
```


```python
print('prediction:')
list(encoder.inverse_transform(ran_fit.predict(x_t)))
```

    prediction:





    ['MotoGP']




```python
print('true topic:')
data_new['article_topic'].loc[355]
```

    true topic:





    'MotoGP'



# - MODEL

Models for text classification I used were MultinomialNB and SGDClassifier because MultinomialNB works well with discrete features such as word counts and SGDClassifier works well with data represented as dense. For this experiment, SGDClassifier with no paramater used got a higher accuracy score, therefore I  did hyperparameter tuning to get a better SGDClassifier model by trying a pair of parameters and finding the optimal one.

# - VALIDATION

I used accuracy because it indicated how good a model was to predict data correctly and I used classification report which also displayed f1 score, recall, prediction, and support (the actual number of occurrences of each class in data we predict) because it made us easier to understand the model perfomance and how good the model predicted some data dispersively. For example, in the classification report of MultinomialNB, it revealed that topic 'Horor' got 0 for f1 score, recall and prediction which meant that the model failed to predict all horor articles, whereas the number of horror articles (support) was 4.

# - FUTURE RESEARCH

Stemming in Bahasa Indonesia is hard, because the data of possible affix combinations and root forms of words is found limited and we may face some problems, such as word sense ambiguity, for example, the word 'berikan' can be chopped as 'ber-i-kan' ('i' will be stored as the root form of word) or 'beri-kan' ('beri' will be extracted) or 'ber-ikan' ('ikan' will be returned instead) or for this case, 'belasan' will be stored as 'bas'. Another example, we can't identify name such as 'Aqilah' (it will be truncated as 'Aqil-ah' and 'Aqil' will be extracted) or in this case, 'Mekkah' as 'Mek'. So for the next research, we suppose to do correction manually for words that are falsely stated from the stemming process. Manually here means creating a function where we would define some root forms of words that may be inaccessible in sastrawi library and manually describe the root forms of the words instead. We can also try boosting method to improve accuracy score 
