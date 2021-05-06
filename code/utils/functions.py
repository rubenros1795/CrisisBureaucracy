import enum
import re, string,os
from glob import glob as gb
import pandas as pd
from collections import Counter
from tqdm import tqdm
from datetime import datetime, timedelta, date
from collections import OrderedDict
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns
import json
from gensim.models import KeyedVectors
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import glob
import os
import string
import numpy
import re
import pandas as pd
import gensim
import random
from tqdm import tqdm
from gensim.models import KeyedVectors
from numpy import vstack,array
from numpy.random import rand
from scipy.cluster.vq import kmeans,vq
import numpy as np
import math
import re
import polyglot
from polyglot.text import Text
import io
import requests
import itertools
from nltk.corpus import stopwords 


base_path = "/media/ruben/Elements/PhD/data/hansard/"


class utils():
    def month_generator(start_month,end_month):
        dates = [start_month, end_month]
        start, end = [datetime.strptime(_, "%Y-%m") for _ in dates]
        return list(OrderedDict(((start + timedelta(_)).strftime(r"%Y-%m"), None) for _ in range((end - start).days)).keys())

    def day_generator(start_day,end_day):
        dates = [start_day, end_day]
        start, end = [datetime.strptime(_, "%Y-%m-%d") for _ in dates]
        return list(OrderedDict(((start + timedelta(_)).strftime(r"%Y-%m-%d"), None) for _ in range((end - start).days)).keys())
    
    def preprocess_(list_txt,lowercase=True,tokenize=False,remove_punc=True,stopwords=[]):
        if lowercase == True:
                list_txt = [str(v).lower() for v in list_txt]
        if remove_punc == True:
            list_txt = [re.sub('[%s]' % re.escape(string.punctuation), '', str(v)) for v in list_txt]
        if tokenize == True:
            list_txt = [str(v).split(' ') for v in list_txt]
        list_txt = [re.sub(' +', ' ', x) for x in list_txt]
        #list_txt = [" ".join([w for w in t if w not in set(stopwords)]) for t in list_txt]
        return list_txt 

    def windowizer(data,words=[],window=5,id_column="",text_column=""):
        result = []
        data = data.reset_index(drop=True)
        for c,text in enumerate(data[text_column]):
            text = str(text).split(' ')
            indices = [c for c,i in enumerate(text) if i in set(words)]
            for c_,ind_ in enumerate(indices):
                left = ind_- window
                right = ind_ + window
                if left < 0:
                    left = 0
                if right > len(text):
                    right = len(text)
                result.append([f"{data[id_column][c]}-{c_}"," ".join(text[left:right])])
        return pd.DataFrame(result,columns=['id','window'])

class data_loader():
    # Class for loading data
    # able to load full dataset for every language 
    # or subsets based on date/words

    def file(fn):
        if os.path.exists(fn):
            return pd.read_csv(fn,sep='\t')
        else:
            print("no DataFrame found")

    def full(data_version="lemmatized_pm"):

        files_path = os.path.join(base_path,data_version)
        list_files = gb(files_path + "*")
        data = pd.DataFrame()
        for f in list_files:
            tdf = pd.read_csv(f,sep='\t')
            data = data.append(tdf)
        return data.reset_index(drop=True)

    def subset(data_version="lemmatized_pm",start_date="",end_date="",words=[]):
        if len(str(start_date)) == 10:
            periods = utils.day_generator(start_date,end_date)
        if len(str(start_date)) == 7:
            periods = utils.month_generator(start_date,end_date)
        if len(str(start_date)) == 4:
            periods = [str(x) for x in range(start_date,end_date)]

        files_path = os.path.join(base_path,data_version) + "/*"
        test_file = pd.read_csv(gb(files_path)[0],sep='\t')

        if start_date == 1957 and end_date == 1985:
            grp = "egrep -iE '" + "|".join(words) + "' " + files_path
            output = subprocess.check_output(grp,shell=True).decode('utf-8')
            output = [l.split('\t') for l in output.split('\n')]
            data = pd.DataFrame(output)
        else:
            data = pd.DataFrame()
            for p in periods:
                try:
                    grp = "egrep -iE '" + "|".join(words) + "' " + files_path.replace('*',f'*{p}*')
                    output = subprocess.check_output(grp,shell=True).decode('utf-8')
                    output = [l.split('\t') for l in output.split('\n')]
                    # print(grp,len(output))
                    if len(output) > 0:
                        output = pd.DataFrame(output)
                        data = data.append(output)
                except Exception as e:
                    print(e)
                    continue
        if len(data) == 0:
            print("empty dataframe, check query")
            return

        data.columns = test_file.columns
        data = data.dropna().reset_index(drop=True)
        data = data.dropna()

        for word in words:     
            data['text_lemmatized'] = [t.replace(word,word.replace(' ','_')) for t in data['text_lemmatized']]
            word_ = word.replace(' ','_')
            data[word+"_hits"] = [len([w for w in t.split(' ') if w == word_]) for t in data['text_lemmatized']]
        return data
    
class frequency():
    def information(data,period_format="month"):
        if period_format == "year":
            data['date'] = [re.search(r'.d.\d{4}-', text).group()[3:-1] for text in data['id']]
        if period_format == "month":
            data['date'] = [re.search(r'.d.\d{4}-', text).group() for text in data['id']]
        if period_format == "day":
            data['date'] = [re.search(r'\d{4}-\d{2}-\d{2}', text).group() for text in data['id']]
        
        return data

    def distribution(freq_info,metadata_selectors):
        freq_info = freq_info[["date"] + metadata_selectors + [c for c in freq_info.columns if "hits" in c]]
        return freq_info.groupby(['date'] + metadata_selectors).sum().reset_index()

    def normalization(freqdata,total_file):
        df_tt = pd.read_csv(total_file)
        df_tt = dict(zip(df_tt.iloc[:,0],df_tt.iloc[:,1]))
        freqdatac = [x for x in freqdata.columns if "_hits" in x]

        for x in freqdatac:
            freqdata[x] = [i / df_tt[int(freqdata['date'][c])] for c,i in enumerate(freqdata[x])]
        return freqdata

    def pmi_windows(text,words=[],window=5):
        list_windows = []
        text = text.split(' ')
        indices = [c for c,i in enumerate(text) if i in set(words)]
        for ind_ in indices:
            left = ind_- window
            right = ind_ + window
            if left < 0:
                left = 0
            if right > len(text):
                right = len(text)
            list_windows.append(text[left:right])
        return list_windows

    def get_pmi_table(data,date,window,word,text_column="text_lemmatized"):
        stopwords_ = list(set(stopwords.words('english')))
        data = data[data['id'].str.contains(str(date))]
        data[text_column] = data[text_column].astype(str)

        all_words = [i.split(' ') for i in data[text_column]]
        all_words = Counter([item for sublist in all_words for item in sublist])
        N = sum(all_words.values())

        windows = [frequency.pmi_windows(str(x),words=[word],window=window) for x in data[text_column]]
        windows = [set(item) for sublist in windows for item in sublist if len(sublist) > 0]

        candidates = set([item for sublist in windows for item in sublist if item not in stopwords_ and len(item) > 3 and '-' not in item and all_words[item] > 5])

        d = []
        for candidate in candidates:

            p1 = all_words[word]
            p2 = all_words[candidate]
            p12 = len([x for x in windows if candidate in x and word in x])

            if p12 > 0:
                try:
                    pmi_reg = math.log(((p12) / (p1 * p2)), 2)
                    pmi_2 = math.log(((p12 ** 2) / (p1 * p2)), 2)
                    pmi_3 = math.log(((p12 ** 3) / (p1 * p2)), 2)
                    npmi = pmi_reg / - math.log(p12)
                    d.append([candidate,p1,p2,p12,pmi_reg,pmi_2,pmi_3,npmi])
                except:
                    continue

        d = pd.DataFrame(d,columns=['w','p1','p2','p12','pmi_reg','pmi_2','pmi_3','npmi'])
        return d
    
    def get_pmi_score(word1,word2,start_year,end_year,text_column):
        data_path = "/media/ruben/Elements/PhD/data/hansard/lemmatized_pm"
        p1 = 0
        p2 = 0
        p12 = 0
        for year in range(int(start_year),int(end_year)+1):
            p1 += int(subprocess.check_output(f'egrep -iE "{word1}"  {data_path}/*{year}* | wc -l',shell=True).decode('utf-8'))
            p2 += int(subprocess.check_output(f'egrep -iE "{word2}"  {data_path}/*{year}* | wc -l',shell=True).decode('utf-8'))
            p12 += int(subprocess.check_output(f'egrep -iE "{word1}.*{word2}|{word2}.*{word1}"  {data_path}/*{year}* | wc -l',shell=True).decode('utf-8'))
        print(f"p1: {p1}, p2: {p2}, p12: {p12}")
        if p12 > 0:
            try:
                pmi_reg = math.log(((p12) / (p1 * p2)), 2)
                pmi_2 = math.log(((p12 ** 2) / (p1 * p2)), 2)
                pmi_3 = math.log(((p12 ** 3) / (p1 * p2)), 2)
                npmi = pmi_reg / - math.log(p12)
                return npmi
            except:
                return "na"

class cluster():
    def generate_matrix(list_words,model_name):
        model = KeyedVectors.load(model_name)
        vocab = set(list(model.wv.vocab))
        list_words = [w for w in set(list_words) if w in vocab]
        print("created list with " + str(len(list_words)) + " words")

        total_list = list()
        
        for word in list_words:
            
            list_word = list()
            
            for term in list_words:
                tmp = model.similarity(word, term)
                list_word.append(tmp)
            
            total_list.append(list_word)
        df = pd.DataFrame(total_list, columns = list_words, index = list_words)
        return df

    def cluster_word(matrix, k):
        centroids,_ = kmeans(matrix,k)
        idx,_ = vq(matrix,centroids)
        
        return dict(zip(list(matrix.index), idx))

class embeddings():
    def train_diachronic(list_bins,sample_size,output_dir = "", min_count=25, workers=6, iter=10, size = 100, window = 15):

        for count,time_ in enumerate(list_bins):

            if count == 0:
                data = data_loader.subset(data_version="lemmatized_pm",start_date=time_[0],end_date=time_[1] + 1,words=[],exact_match=True,preprocess=True)
                if len(data) > sample_size:
                    data = data.sample(sample_size).reset_index(drop=True)
                texts = list(data['text_lemmatized'])
                texts = [t.split(' ') for t in texts]
                print(time_[0],'-',time_[1],len(texts))
                #model = gensim.models.Word2Vec(texts, min_count=min_count, workers=workers, iter=iter, size = size, window = window)
                #model.wv.save_word2vec_format(os.path.join(f"{output_dir}/{time_[0]}-{time_[1]}-model.bin"), binary=True)
            if count != 0:
                data = data_loader.subset(data_version="lemmatized_pm",start_date=time_[0],end_date=time_[1] + 1,words=[],exact_match=True,preprocess=True)
                if len(data) > sample_size:
                    data = data.sample(sample_size).reset_index(drop=True)
                texts = list(data['text_lemmatized'])
                texts = [t.split(' ') for t in texts]
                #model.build_vocab(texts, update=True)
                #model.train(texts, total_examples = model.corpus_count, start_alpha = model.alpha, end_alpha = model.min_alpha, epochs = model.iter)
                #model.wv.save_word2vec_format(os.path.join(f"{output_dir}/{time_[0]}-{time_[1]}-model.bin"), binary=True)
                print(time_[0],'-',time_[1],len(texts))

    def train(data,sample_size=100000,min_count=25, workers=6, iter=10, size = 100, window = 12):
        if sample_size > len(data):
            data = data.sample(sample_size)
        texts = [t.split(' ') for t in data['text']]
        model = gensim.models.Word2Vec(texts, min_count=min_count, workers=workers, iter=iter, size = size, window = window)
        model.wv.save_word2vec_format(os.path.join(f"/media/ruben/OSDisk/Users/ruben.ros/Documents/GitHub/CrisisBureaucracy/results/w2v-models/single-model-{sample_size}.bin"), binary=True)
        return model

    
    # def DataFrameMostSimilar(language,search_term,words,topn=15):
    #     d = pd.DataFrame()
    #     print("searching in",f'/media/ruben/OSDisk/Users/ruben.ros/Documents/GitHub/ParlaMintCase/results/models/{language}/*{"_".join(words)}*')
    #     for i in sorted(gb(f'/media/ruben/OSDisk/Users/ruben.ros/Documents/GitHub/ParlaMintCase/results/models/{language}/*{"_".join(words)}*')):
    #         m = KeyedVectors.load_word2vec_format(i,binary=True)
    #         try:
    #             d[i[-17:-10]] = list([x[0] for x in m.wv.most_similar(search_term,topn=topn)])
    #         except Exception as e:
    #             continue

    #     return d

    # def DiachronicSimilarity(language,word1,word2,words,topn=15):
    #     d = []
    #     print("searching in",f'/media/ruben/OSDisk/Users/ruben.ros/Documents/GitHub/ParlaMintCase/results/models/{language}/*{"_".join(words)}*')
        
        
    #     for i in sorted(gb(f'/media/ruben/OSDisk/Users/ruben.ros/Documents/GitHub/ParlaMintCase/results/models/{language}/*{"_".join(words)}*')):
    #         m = KeyedVectors.load_word2vec_format(i,binary=True)
    #         try:
    #             d.append([i[-17:-10],m.wv.similarity(word1,word2)])
    #         except Exception as e:
    #             continue

    #     return pd.DataFrame(d,columns="month similarity".split())

class plotting():

    def style(vl=12,pal='Paired',font='CMU Serif,Bold'):
        sns.set(font=font,rc={'axes.axisbelow': True,'axes.edgecolor': 'lightgrey','axes.facecolor': 'None', 'axes.grid': True,'grid.color':'whitesmoke','axes.labelcolor':'black','axes.spines.top': True,'figure.facecolor': 'white','lines.solid_capstyle': 'round','patch.edgecolor': 'w','patch.force_edgecolor': True,'text.color': 'black','xtick.bottom': False,'xtick.color': 'black','xtick.direction': 'out','xtick.top': False,'ytick.color': 'black','ytick.direction': 'out','ytick.left': False, 'ytick.right': False})
        sns.set_context("notebook", rc={"font.size":20,"axes.titlesize":26, "axes.labelsize":20})
        sns.set_palette(pal,vl)
        
class DenseTfIdf(TfidfVectorizer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def transform(self, x, y=None) -> pd.DataFrame:
        res = super().transform(x)
        df = pd.DataFrame(res.toarray(), columns=self.get_feature_names())
        return df

    def fit_transform(self, x, y=None) -> pd.DataFrame:
        res = super().fit_transform(x, y=y)
        #df = pd.DataFrame(res.toarray(), columns=self.get_feature_names(), index=x.index())
        return self,res

class tfidf():

    def get_docterms(data,text_column,**kwargs):
        texts = list(data[text_column])
        return DenseTfIdf(sublinear_tf=True, max_df=0.5,min_df=0.052,encoding='ascii',lowercase=True,stop_words='english',**kwargs).fit_transform(texts)

    def get_topterms(tfidf_object,docterms,data,category_column):
        docterms = pd.DataFrame(docterms.toarray(), columns=tfidf_object.get_feature_names(),index=data.index)
        d = pd.DataFrame()

        for cat in set(data[category_column]):
            # Need to keep alignment of indexes between the original dataframe and the resulted documents-terms dataframe
            df_class = data[(data[category_column] == cat)]
            df_docs_terms_class = docterms.iloc[df_class.index]
            # sum by columns and get the top n keywords
            dfop = df_docs_terms_class.sum(axis=0).nlargest(n=100)
            dfop = pd.DataFrame(dfop).reset_index()
            dfop.columns = [cat + " terms",cat + " score"]
            dfop[cat + " score"] = [round(x,2) for x in dfop[cat + " score"]]

            d[cat] = dfop[cat + " terms"]
        return d
