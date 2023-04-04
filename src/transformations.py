# loading packages
import os
import pandas as pd
import numpy as np
import re

# nltk
import nltk

nltk.download('punkt')

# stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

english_stop_words = stopwords.words('english')

# tokenizing
from nltk.tokenize import word_tokenize

# normalizing
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

current_path = os.getcwd()

# remove underscore
def remove_underscore(text):
    return re.sub('_', ' ', text)


# convert to lower case
def to_lowercase(text):
    return text.lower()


# remove html tags
def remove_html(text):
    return re.sub(r'<[^>]+>', ' ', text)


# remove punctuation
def remove_punc(text):
    res = re.sub(r'\'', '', text)  # apostrophe removed without splitting word
    res = re.sub(r'[^\w\s]', ' ', res)
    return res


# remove numbers
def remove_num(text):
    return re.sub(r'\w*\d\w*', ' ', text)


# remove extra spaces
def remove_whitespace(text):
    return re.sub(r' {2,}', ' ', text).strip()


# remove stopwords
def remove_stopwords(text):
    text_tokens = word_tokenize(text)
    tokens_without_sw = [word for word in text_tokens if not word in english_stop_words]
    return ' '.join(tokens_without_sw)


# stemming
def stem_text(text):
    tokens = [stemmer.stem(word) for word in text.split()]
    result = ' '.join(tokens)
    return result


def get_cleantext(text, stemming=False):
    """
    Applies all text cleaning steps on the input String. Returns clean text String.
    Stemming is skipped by default, set stemming=True to stem text
    """
    removed_extra_chars = remove_underscore(remove_whitespace(remove_num(remove_punc(remove_html(to_lowercase(text))))))
    res = remove_stopwords(removed_extra_chars)
    if stemming:
        res = stem_text(res)
    return res

# Feature engineering
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import gensim.downloader as api
from gensim.models import KeyedVectors

# 1) bow
def bow(X, ngram_range=(1, 1)):
    """
    ngram_range is set to (1,1) in default to extract only individual words (unigrams)
    can change to (2,2) for bigrams or (1,2) for both ungrams and bigrams
    """
    # Create an instance of the CountVectorizer class
    vectorizer = CountVectorizer(ngram_range=ngram_range)

    # Fit the vectorizer on the text data and transform it into a matrix
    bow_matrix = vectorizer.fit_transform(X)

    X = bow_matrix.toarray()

    return X

# 2) TF_IDF
def tf_idf(X):

    # Create an instance of the TfidfVectorizer class, can modify its parameters such as ngram
    # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    vectorizer = TfidfVectorizer()

    # Fit the vectorizer on the text data and transform it into a matrix
    matrix = vectorizer.fit_transform(X)
    X = matrix.toarray()

    return X

# 3) word2vec
# use pre-trained word2vec model
#wv = api.load('word2vec-google-news-300')
#wv.save('/content/drive/MyDrive/Dsa4263/vectors.kv')
wv = KeyedVectors.load(current_path + 'vectors.kv')


def word2vec(X):
    def get_mean_vector(text, wv):
        """
        numerical representation for the sentence = mean(words in the sentence)
        """
        vector_size = wv.vector_size
        wv_res = np.zeros(vector_size)
        ctr = 0
        for w in text:
            if w in wv:
                ctr += 1
                wv_res += wv[w]
        if ctr == 0:
            return wv_res
        else:
            wv_res = wv_res / ctr
            return wv_res
    x_split = list(map(lambda x: x.split(),X))
    X = list(map(lambda text: get_mean_vector(text,wv), x_split))

    return X

def skl_tfidf(df, col_name='stem_clean_text'):
   """
   Input: df, name of column with text
   """
   texts = df[col_name]
   tfidf_vectorizer = TfidfVectorizer(min_df=3, 
                                      max_df=0.85, 
                                      max_features=5000, 
                                      ngram_range=(1, 2))
   tfidf = tfidf_vectorizer.fit_transform(texts)
   return tfidf, tfidf_vectorizer
