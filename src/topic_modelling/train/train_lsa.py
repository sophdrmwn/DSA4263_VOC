import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

def lsa(df, n):
    """
    This function performs Latent Semantic Analysis (LSA) on a given DataFrame using TruncatedSVD (Singular Value Decomposition) for dimensionality        reduction. 

    Args:
    - df (pandas DataFrame): The input DataFrame for which LSA needs to be performed.
    - n (int): The number of topics to extract from the input DataFrame.

    Returns:
    - topics_df (pandas DataFrame): A DataFrame containing the top terms for each topic extracted from the input DataFrame.
    - topic_labels (numpy array): An array of topic labels for each row in the input DataFrame.
    """
    #build LSA model
    lsa_model = TruncatedSVD(n_components=n)
    lsa_topic_matrix = lsa_model.fit_transform(df)
    terms = df.columns.values
    topics=[]
    #extract top 10 words for each topic
    for i, comp in enumerate(lsa_model.components_):
        terms_comp = zip(terms, comp)
        sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:10]
        wl=[x[0] for x in sorted_terms]
        topics.append(wl)
        print("\n Topic "+str(i)+": ",end=' ')
        for t in sorted_terms:
            print(t[0],end=' ')
    return pd.DataFrame(topics), np.argmax(lsa_topic_matrix, axis=1)





    
    
    
    
    
