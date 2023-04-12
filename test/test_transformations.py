from src.trainsformations import *
import numpy as np


# test simple cleaning functions
def test_remove_underscore():
    text = "hello_world"
    expected_output = "hello world"
    assert remove_underscore(text) == expected_output

def test_to_lowercase():
    text = "Hello World"
    expected_output = "hello world"
    assert to_lowercase(text) == expected_output

def test_remove_html():
    text = "<p>Hello World</p>"
    expected_output = " Hello World "
    assert remove_html(text) == expected_output

def test_remove_punc():
    text = "Hello! World's? Punctuation."
    expected_output = "Hello Worlds Punctuation"
    assert remove_punc(text) == expected_output

def test_remove_num():
    text = "Hello 123 World 456"
    expected_output = "Hello World"
    assert remove_num(text) == expected_output

def test_remove_whitespace():
    text = "  Hello  World  "
    expected_output = "Hello World"
    assert remove_whitespace(text) == expected_output

def test_remove_stopwords():
    text = "the quick brown fox jumps over the lazy dog"
    expected_output = "quick brown fox jumps lazy dog"
    assert remove_stopwords(text) == expected_output

def test_stem_text():
    text = "jumps jumping jumped"
    expected_output = "jump jump jump"
    assert stem_text(text) == expected_output

def test_get_cleantext():
    text = "Hello <b>World!</b> 123."
    expected_output = "hello world"
    assert get_cleantext(text) == expected_output

# test feature engineering functions
def test_bow():
    X = ["This is a sentence.", "This is another sentence."]
    expected_output = [[1, 1, 1, 0, 0], [1, 1, 0, 1, 0]]

    # Test with both unigrams and bigrams
    output = bow(X, ngram_range=(1,2))
    expected_output = [[1, 1, 1, 0, 0, 1, 0], [1, 1, 0, 1, 1, 0, 1]]
    assert output == expected_output, f"Expected {expected_output}, but got {output}"

def test_tf_idf():
    # Create some test data
    X = ['This is a test', 'Another test string', 'Yet another test string']
    # Call the function to obtain the feature matrix and the vectorizer
    X_transformed, vectorizer = tf_idf(X)
    # Check that the feature matrix has the expected dimensions
    assert X_transformed.shape == (3, 4)

def test_get_mean_vector():
    wv_dict = {'hello': [1,2,3,4,5], 'world': [5,4,3,2,1]}
    wv = KeyedVectors(vector_size=5)
    wv.add_vectors(list(wv_dict.keys()), list(wv_dict.values()))
    
    text = ['hello', 'world']
    vector = get_mean_vector(text, wv)
    expected_vector = np.array([3., 3., 3., 3., 3.])
    assert np.allclose(vector, expected_vector)

def test_word2vec():
    wv_dict = {'hello': [1,2,3,4,5], 'world': [5,4,3,2,1]}
    wv = KeyedVectors(vector_size=5)
    wv.add_vectors(list(wv_dict.keys()), list(wv_dict.values()))
    
    X = ['hello world', 'this is a test']
    result = word2vec(X, wv=wv)
    expected_result = np.array([[3., 3., 3., 3., 3.], [0., 0., 0., 0., 0.]])
    
    assert np.allclose(result, expected_result)

def test_new_bow():
    X = ["This is the first document.", "This is the second document."]
    expected_output = pd.DataFrame({
        'document': [1, 1],
        'first': [1, 0],
        'is': [1, 1],
        'second': [0, 1],
        'the': [1, 1],
        'this': [1, 1]
    })
    assert new_bow(X).equals(expected_output)

def test_new_tfidf():
    X = ["This is the first document.", "This is the second document.", "And this is the third."]
    expected_output = pd.DataFrame({
        'and': [0.0, 0.0, 0.693147],
        'document': [0.438776, 0.438776, 0.438776],
        'first': [0.693147, 0.0, 0.0],
        'is': [0.438776, 0.438776, 0.438776],
        'second': [0.0, 0.693147, 0.0],
        'the': [0.438776, 0.438776, 0.438776],
        'third': [0.0, 0.0, 0.693147],
        'this': [0.438776, 0.438776, 0.438776]
    })
    assert new_tfidf(X).round(6).equals(expected_output.round(6))

# unit test for lemmatize_text function   
def test_get_wordnet_pos():
    assert get_wordnet_pos('good') == wordnet.ADJ
    assert get_wordnet_pos('book') == wordnet.NOUN
    assert get_wordnet_pos('run') == wordnet.VERB
    assert get_wordnet_pos('quickly') == wordnet.ADV
    assert get_wordnet_pos('hello') == wordnet.NOUN


def test_lemmatize_text():
    assert lemmatize_text('dogs running fast') == 'dog run fast'
    assert lemmatize_text('I am playing football') == 'I be play football'
    assert lemmatize_text('The books are on the table') == 'The book be on the table'
    assert lemmatize_text('She is singing beautifully') == 'She be sing beautifully'
    assert lemmatize_text('He was walking slowly') == 'He be walk slowly'

# unit test for words_remove function
def test_words_remove():
    assert words_remove('This is one good book') == 'This good book'
    assert words_remove('I want to try this product') == 'I want to this product'
    assert words_remove('I ordered too much food') == 'I ordered food'
    assert words_remove('The cat came running') == 'The cat came running'
    assert words_remove('This is an Amazon product') == 'This is an product'

# unit test for select_pos_tag function
def test_select_pos_tag():
    # create a sample dataframe
    df = pd.DataFrame({
        'apples': [0.5, 0.1, 0.2],
        'eat': [0.2, 0.3, 0.1],
        'delicious': [0.1, 0.2, 0.3],
        'run': [0.3, 0.1, 0.4]
    })
    
    # select nouns and verbs only
    pt = ['n', 'v']
    expected = pd.DataFrame({
        'apples': [0.5, 0.1, 0.2],
        'eat': [0.2, 0.3, 0.1],
        'run': [0.3, 0.1, 0.4]
    })
    
    # ensure only the columns with the selected POS tags are returned
    assert select_pos_tag(df, pt).equals(expected)

