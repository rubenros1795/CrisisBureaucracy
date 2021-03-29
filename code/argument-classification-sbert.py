# Argument Classification using Sentence-Bert

import os
import pandas as pd 
from sentence_transformers import SentenceTransformer
import scipy.spatial.distance
import scipy
import nltk
from nltk.corpus import stopwords
from classification import *
from utils.functions import *
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import numpy as np
import swifter

import warnings
warnings.filterwarnings("ignore")

# Source: https://github.com/subhasisj/FastAPI-Streamlit-Docker-NLP/blob/487bd285b2d73509147c88055b3658a28e7d800b/Notebooks/spam-ham-classifier-sentence-bert-pos.ipynb

model = SentenceTransformer('bert-base-nli-mean-tokens')


# Import annotations and match
annotations = pd.read_csv('~/Documents/GitHub/CrisisBureaucracy/data/classifier/annotated-arguments-bureaucracy.csv')
annotations = annotations[["id","label","text"]]
# annotations['metadata'] = ''

# refdata = pd.read_csv('~/Documents/GitHub/CrisisBureaucracy/data/classifier/training_data_full.csv',sep='\t')
# refdata['id-ann'] = [x + 594 for x in refdata.index]

# for c,i in enumerate(annotations['id']):
#     annotations['metadata'][c] = str(refdata[refdata['id-ann'] == i].reset_index(drop=True)['id'][0])


# Preprocess annotated data
stops = stopwords.words('english') + ["hon","member","right","friend","mr",'hon.','make','say','great']
annotations['text'] = utils.preprocess_(annotations['text'])
annotations = annotations.drop('id',axis=1).reset_index(drop=True)
annotations['label'] = [x-1 for x in annotations['label']]


#labels = {0:"neutral",1:"inefficient",2:"powerful/large",3:"centralization",4:"freedom",5:"expensive",6:"anti-democratic"}
#annotations['label'] = annotations['label'].astype(str)

# Build Embeddings
vectors_swifter = annotations['text'].swifter.apply(model.encode)
annotations['sentence-bert'] = vectors_swifter

train_df = annotations.copy()
print(train_df.head(5).iloc[:,:-1])


def stack_embeddings(embeddings):
    return np.vstack(embeddings.values)

ct = ColumnTransformer([
    ('bag of ngrams', TfidfVectorizer(ngram_range=(1, 2), max_features=3000), 'text'),
    ('sentence bert', FunctionTransformer(stack_embeddings), 'sentence-bert')],
    remainder='passthrough')

## Classification
# lm = LogisticRegression()
xgb = XGBClassifier(random_state=0)

# pipeline = Pipeline([('transformer', ct), ('classifier', lm)])
pipeline = Pipeline([('transformer', ct), ('classifier', xgb)])


y,X = train_df.pop('label'),train_df

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.95, random_state=1,stratify=y)

model = pipeline.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(metrics.classification_report(y_test,y_pred))
