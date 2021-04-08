import re, string,os,random
from glob import glob as gb
import pandas as pd
from collections import Counter
from tqdm import tqdm
import subprocess
from utils.functions import *
from gensim.models import keyedvectors

output_dir = "~/Documents/GitHub/CrisisBureaucracy/results/w2v-models"

texts = []
for year in range(1957,1986):
    data = data_loader.subset(data_version="lemmatized_pm",start_date=year,end_date=year+1,words=[],exact_match=True,preprocess=True).reset_index(drop=True)
    t = list(data['text_lemmatized'])
    t = [x.split(' ') for x in t]
    t = random.sample(t,2500)
    texts += t
    
model = gensim.models.Word2Vec(texts, min_count=10, workers=8, iter=12, size = 120, window = 16)
model.wv.save_word2vec_format(os.path.join(f"{output_dir}/model-single-sample.bin"), binary=True)