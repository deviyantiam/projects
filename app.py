import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, Response
import pickle
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import json
from mo import Model, Model2
import re
from customer import RFM, Fin_RFM
from sklearn.neighbors import NearestNeighbors
# import findspark
# findspark.init('/usr/local/Cellar/apache-spark/2.4.0/libexec')
# from pyspark.sql.types import StructType,StructField
# from pyspark.ml.linalg import Vectors
# from pyspark.sql import Row
# from pyspark.ml.clustering import KMeansModel
# from pyspark.ml.feature import MinMaxScalerModel
# from pyspark.sql import SparkSession
# from pyspark.sql.types import IntegerType

# spark = SparkSession.builder \
#     .master("local") \
#     .appName("RFM Analysis") \
#     .config("spark.some.config.option", "some-value") \
#     .getOrCreate()
    
    

regEx = re.compile('[^a-zA-Z]+')
def cleanReviews(reviewText):
    reviewText = regEx.sub(' ', reviewText).strip()
    return reviewText

app = Flask(__name__)
df_movie= pd.read_csv('movies.csv')
df_movie['year'] = df_movie.title.str.extract('(\(\d\d\d\d\))',expand=False)
df_movie['year'] = df_movie.year.str.extract('(\d\d\d\d)',expand=False)
df_movie['title'] = df_movie.title.str.replace('(\(\d\d\d\d\))', '')
df_movie['title'] = df_movie['title'].apply(lambda x: x.strip())
df_movie['genres'] = df_movie.genres.str.split('|')
tit=df_movie['title'].values.tolist()
df = pd.read_json("/Users/devi/Documents/finalproject/recom/data_amazon.json",lines=True)
df[['Helpful','Total_whethernot_helpful']] = pd.DataFrame(df.helpful.values.tolist(), index = df.index)
df.drop_duplicates(subset=['reviewerID', 'asin','unixReviewTime'],inplace=True)
df['Helpful %'] = np.where(df['Total_whethernot_helpful'] > 0, df['Helpful'] / df['Total_whethernot_helpful'], -1)
df['% Upvote'] = pd.cut(df['Helpful %'], bins = [-1, 0, 0.2, 0.4, 0.6, 0.8, 1.0], labels = ['Empty', '0-20%', '20-40%', '40-60%', '60-80%', '80-100%'], include_lowest = True)
df['Id'] = df.index;
df["usefulScore"] = (df["Helpful"]/df["Total_whethernot_helpful"]).apply(lambda n: "useful" if n > 0.8 else "useless")
df=df.dropna(subset=['% Upvote'])
count = df.groupby("asin", as_index=False).count()
county=count['asin'].values.tolist()

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/movie")
def movie():
    return render_template('mov_in.html')

@app.route('/autocomplete',methods=['GET'])
def autocomplete():
    search = request.args.get('autocomplete')
    app.logger.debug(search)
    return Response(json.dumps(tit), mimetype='application/json')

@app.route('/movrec', methods=['POST'])
def recmo():
    films = request.form.getlist("title[]")
    rating_=request.form.getlist("rat[]")
    rat_new=[]
    for i in rating_:
        i=float(i)
        rat_new.append(i)
    data_tuples = list(zip(films,rat_new))
    input_mo=pd.DataFrame(data_tuples, columns=['title','rating'])
    hasil_gen=Model.bygenre(input_mo)
    hasil_user=Model2.byusers(input_mo)
    # with open("/pro_pur/templates/res_mov.html", 'w') as _file:
    #     _file.write(input_mo.to_html() + "\n\n" + hasil.to_html())
    return render_template('mov_ans.html',tables=[hasil_gen.to_html(classes='data')], titles=hasil_gen.columns.values,tably=[hasil_user.to_html(classes='data')],titly=hasil_user.columns.values)

@app.route('/product')
def product():
    return render_template('pro.html')

@app.route('/autoc',methods=['GET'])
def autoc():
    search = request.args.get('autocomplete')
    app.logger.debug(search)
    return Response(json.dumps(county), mimetype='application/json')

@app.route('/rec_goods',methods=['POST'])
def rec_goods():
    nei=pickle.load(open('pro_rec.pkl','rb'))
    review_ = request.form['revx']
    review_=cleanReviews(review_)
    prox = request.form['prodx']
    rat_x = request.form['ratx']
    dfMerged = pd.merge(df, count, how='right', on=['asin'])
    dfMerged["totalReviewers"] = dfMerged["reviewerID_y"]
    dfMerged["overallScore"] = dfMerged["overall_x"]
    dfMerged["summaryReview"] = dfMerged["summary_x"]
    dfMerged = dfMerged.sort_values(by='totalReviewers', ascending=False)
    dfCount = dfMerged[dfMerged.totalReviewers >= 80]
    dfProductReview = df.groupby("asin", as_index=False).mean()
    ProductReviewSummary = dfCount.groupby("asin")["summaryReview"].apply(str)
    ProductReviewSummary = pd.DataFrame(ProductReviewSummary)
    df3 = pd.merge(ProductReviewSummary, dfProductReview, on="asin", how='inner')
    del dfProductReview
    del dfMerged
    df3 = df3[['asin','summaryReview','overall']]
    inputy=[review_]
    countVector=pickle.load(open('pro_countv.pkl','rb'))
    df_inputy=countVector.transform(inputy)
    df_inputy=pd.DataFrame(df_inputy.A, columns=countVector.get_feature_names())
    di,iy=nei.kneighbors(df_inputy)
    ix=iy[0]
    pro1=df3['asin'][ix[0]]
    pro2=df3['asin'][ix[1]]
    pro3=df3['asin'][ix[2]]
    r1=df3['overall'][ix[0]]
    r2=df3['overall'][ix[1]]
    r3=df3['overall'][ix[2]]
    convrat=pickle.load(open('pro_crat.pkl','rb'))
    testCounts = convrat.transform(inputy)
    tfidf_transformer=pickle.load(open('pro_tfidrat.pkl','rb'))
    testTfidf = tfidf_transformer.transform(testCounts)
    model_=pickle.load(open('pro_lograt.pkl','rb'))
    result = model_.predict(testTfidf)[0]
    probability = model_.predict_proba(testTfidf)[0]
    re_dict={1:'Positive Review',0:'Negative Review',3:'Neutral'}
    feedb=re_dict[result]
    pron=probability[0]
    probb=probability[1]
    print("The feedback estimated as %s: negative prob %f, positive prob %f" % (re_dict[result], probability[0], probability[1]))
    return render_template('pro_ans.html',feedb=feedb,pron=pron,probb=probb,r1=r1,r2=r2,r3=r3,pro1=pro1,pro2=pro2,pro3=pro3)

@app.route("/customer")
def cust():
    return render_template('rfm_kmeans.html')

# @app.route("/cust_seg",methods=['POST'])
# def cust_seg():
#     rece= int(request.form['recency'])
#     fre = int(request.form['freq'])
#     mon = int(request.form['mone'])
#     re_rec=str(RFM.RScore(rece))
#     re_f=str(RFM.FScore(fre))
#     re_mon=str(RFM.MScore(mon))
#     rfm_s=re_rec+re_f+re_mon
#     rfmscore=Fin_RFM.XScore(rfm_s)
#     scll=MinMaxScalerModel.load('filepath/scaling')
#     kmll=KMeansModel.load('filepath/rfmmodel')
#     schema_ = StructType([StructField('r', IntegerType()), StructField('f',IntegerType()),StructField('m', IntegerType())])
#     rows_ = [Row(r=rece, f=fre, m=mon)]
#     inp_ = spark.createDataFrame(rows_, schema_)
#     inp_=inp_.rdd.map(lambda r: (Vectors.dense(r[0:]),)).toDF(['rfm'])
#     sca_inp=scll.transform(inp_)
#     km_inp=kmll.transform(sca_inp)
#     df_inp=km_inp.select('*').toPandas()
#     idx_c=df_inp.loc[0,'prediction']
#     c_label={0:'Champions',1:'Acquaintances',2:'Disengaged'}
#     ans_kmean=c_label[idx_c]
#     return render_template('cus_ans.html',re_mon=rfmscore,kmean_c=ans_kmean)

@app.route('/news')
def berita():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    model = pickle.load(open('model_news.pkl','rb'))
    cv_=pickle.load(open('news_cv.pkl','rb'))
    enc_=pickle.load(open('enc_news.pkl','rb'))
    y_for_test = request.form['news_']
    y_for_test=pd.Series(y_for_test)
    factory = StopWordRemoverFactory()
    stopword = factory.create_stop_word_remover()
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    xy=[]
    for i in y_for_test.values:
        stops_=stopword.remove(stemmer.stem(i))
        wordy_=''
        for st in stops_.split(' '):
            if st.isalpha():      
                wordy_+=st+' '
        xy.append(wordy_)
    x_t=cv_.transform(xy)
    resu=model.predict(x_t)
    print('prediction:')
    s = [str(i) for i in list(enc_.inverse_transform(resu))] 
    res = ", ".join(s)   
    return render_template('index.html', prediction_text='Topiknya adalah {}. Ya kan?'.format(res))
    
if __name__ == "__main__":
    app.run(debug=True)