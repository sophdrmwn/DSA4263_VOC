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

# loadind raw data
current_path = os.getcwd()
df = pd.read_csv(current_path + '/data/reviews.csv', encoding='unicode_escape')

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
  res = re.sub(r'\'', '', text) # apostrophe removed without splitting word
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
  removed_extra_chars =  remove_underscore(remove_whitespace(remove_num(remove_punc(remove_html(to_lowercase(text))))))
  res = remove_stopwords(removed_extra_chars)
  if stemming:
    res = stem_text(res)
  return res

# clean raw data
df['clean_text'] = df['Text'].apply(lambda x: get_cleantext(x))
df['stem_clean_text'] = df['Text'].apply(lambda x: get_cleantext(x, stemming=True))

# save clean data to csv
df.to_csv(current_path + '/data/clean_reviews.csv', index=False)