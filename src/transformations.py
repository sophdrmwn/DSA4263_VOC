# loading packages
import os
import pandas as pd
import numpy as np
import re
import pickle
import gensim
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import gensim.downloader as api
from gensim.models import KeyedVectors

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
root_path = os.path.dirname(current_path)

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

# 1) TF_IDF
def tf_idf(X, vectorizer = None):
    X = list(map(lambda x: get_cleantext(x),X))
    if not vectorizer:
        # Create an instance of the TfidfVectorizer class, can modify its parameters such as ngram
        vectorizer = TfidfVectorizer()
        # Fit the vectorizer on the text data and transform it into a matrix
        matrix = vectorizer.fit(X)
    
    # Transform the input data into a matrix using the trained vectorizer
    matrix = vectorizer.transform(X)
    X = matrix.toarray()

    return X, vectorizer

# 2) word2vec
# use pre-trained word2vec model
# wv = api.load('word2vec-google-news-300')
#wv.save('/content/drive/MyDrive/Dsa4263/vectors.kv')
#wv = KeyedVectors.load(current_path + 'vectors.kv')

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
        
def word2vec(X, wv_name='word2vec-google-news-300'):

    wv = api.load(wv_name)

    x_clean = list(map(lambda x: get_cleantext(x, ),X))
    x_split = list(map(lambda x: x.split(),x_clean))
    X_list = list(map(lambda text: get_mean_vector(text,wv), x_split))
    
    X = np.array(X_list)
    return X

# 3) Bag of Words
def new_bow(X, ngram_range=(1, 1)):
    vectorizer = CountVectorizer(ngram_range=ngram_range)
    bow_matrix = vectorizer.fit_transform(X)
    df_bow = pd.DataFrame(bow_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    return df_bow

# 4) TF_IDF with save option and without cleaning
def new_tfidf(X, save=False, ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None):
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, 
                                 max_df=max_df, 
                                 min_df=min_df, 
                                 max_features=max_features)
    matrix = vectorizer.fit_transform(X)
    df_tfidf = pd.DataFrame(matrix.toarray(), columns=vectorizer.get_feature_names_out())
    # save tfidf vectorizer if needed
    if save:
        pickle.dump(vectorizer, open(root_path+"/models/tfidfvectorizer.pickle", "wb"))
    return df_tfidf

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

#lemmatization
def get_wordnet_pos(word):
  """Map POS tag to first character lemmatize() accepts"""
  tag = nltk.pos_tag([word])[0][1][0].lower()
  tag_dict = {"j": wordnet.ADJ,
              "n": wordnet.NOUN,
              "v": wordnet.VERB,
              "r": wordnet.ADV}
  return tag_dict.get(tag, wordnet.NOUN)

def lemmatize_text(text):
  lemmatizer = WordNetLemmatizer()
  tokens = [lemmatizer.lemmatize(word, pos=get_wordnet_pos(word)) for word in text.split()]
  result = ' '.join(tokens)
  return result

#further cleaning
def words_remove(text):
  ls = ['one','get','use','try','much','go','amazon','even','also','give','add','say','come','order','like']
  tokens = [word for word in text.split() if word not in ls]
  result = ' '.join(tokens)
  return result

#select certain types of words like nouns, adjectives...
def select_pos_tag(df, pt=['j','n','v','r']):
  col = df.columns.values.tolist()
  new_col = filter(lambda c: nltk.pos_tag([c])[0][1][0].lower() in pt, col)
  return df.loc[:,new_col]