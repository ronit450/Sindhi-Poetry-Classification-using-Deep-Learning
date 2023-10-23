from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import pandas as pd
import numpy as np

class Token:
    def __init__(self, data):
        self.data = self.df_Maker(data)
    

    def df_Maker(self, data):
        df =  pd.read_csv(data)
        print(df.head())
        return df
    
    def Tf_IDF(self):
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(self.data)
    
    def word_embeddings(self):
        poetry_sentences = [couplet.split() for couplet in self.data['couplets']]
        model = Word2Vec(poetry_sentences, vector_size=100, window=5, min_count=1, sg=0)
        word_embeddings = [model.wv[word] for word in poetry_sentences.split() if word in model.wv]

a = Token('data/dataCompiled.csv')
