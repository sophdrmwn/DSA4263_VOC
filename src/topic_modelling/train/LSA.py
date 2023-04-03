import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD

#lemmatization
def get_wordnet_pos(word):
  """Map POS tag to first character lemmatize() accepts"""
  tag = nltk.pos_tag([word])[0][1][0].lower()
  tag_dict = {"a": wordnet.ADJ,
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

#feature engineering
def bow(df, ngram_range = (1,1)): 

  clean_text = df['clean_text'].tolist()
 
  # Create an instance of the CountVectorizer class
  vectorizer = CountVectorizer(ngram_range = ngram_range)

  # Fit the vectorizer on the text data and transform it into a matrix
  bow_matrix = vectorizer.fit_transform(clean_text)

  # Convert the matrix to a pandas dataframe
  # Note: if scikit-learn < 1.0, use get_feature_names()
  df_bow = pd.DataFrame(bow_matrix.toarray(), columns=vectorizer.get_feature_names_out())

  return df_bow,vectorizer

def tf_idf(df):
  clean_text = df['clean_text'].tolist()
  vectorizer = TfidfVectorizer()

  # Fit the vectorizer on the text data and transform it into a matrix
  matrix = vectorizer.fit_transform(clean_text)

  # Convert the matrix to a pandas dataframe
  df_tfidf = pd.DataFrame(matrix.toarray(), columns=vectorizer.get_feature_names_out())

  return df_tfidf, vectorizer

from sklearn.decomposition import TruncatedSVD

def lsa(df, vec, n):
  lsa_model = TruncatedSVD(n_components=n)
  lsa_topic_matrix = lsa_model.fit_transform(df)
  terms = vec.get_feature_names_out()
  topics=[]
  for i, comp in enumerate(lsa_model.components_):
    terms_comp = zip(terms, comp)
    sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:10]
    wl=[x[0] for x in sorted_terms]
    topics.append(wl)
    print("\n Topic "+str(i)+": ",end=' ')
    for t in sorted_terms:
        print(t[0],end=' ')
  return pd.DataFrame(topics)

def lsa_unit_testing(topics):
    if topics.shape==(n,10) and not(topics.isnull().values.any()):
        return True
    else:
        return False
