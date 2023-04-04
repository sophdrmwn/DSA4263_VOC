import pandas as pd
from sklearn.decomposition import TruncatedSVD

def lsa(df, n):
  lsa_model = TruncatedSVD(n_components=n)
  lsa_topic_matrix = lsa_model.fit_transform(df)
  terms = df.columns.values
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



    
    
    
    
    
