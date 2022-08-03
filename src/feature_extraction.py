# %%
from numpy.core.fromnumeric import shape
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from collections import namedtuple

df = pd.read_csv('../dataset/Reviews.csv', usecols=['Text', 'Score'],index_col=False)

# We define the scores as: 1,2 negative , 3 neutral , 4,5 positive

# Score 1,2 means negative
df.Score[df.Score < 3] = -1
# Score 4,5 is positive
df.Score[df.Score > 3] = 1
# Score 3 is neutral
df.Score[df.Score == 3] = 0

# sample equal for each class (neutral are the lowest in count)
sample_size = 42640
df_neg = df[df.Score == -1].sample(sample_size,random_state=30)
df_neut = df[df.Score == 0].sample(sample_size,random_state=30)
df_pos = df[df.Score == 1].sample(sample_size,random_state=30)
# take equal size of each class
df = pd.concat([df_neg, df_neut, df_pos])

def extractTFIDF():
    '''
    produce a new dataset where the text field has transformed into features
    '''
    Y = df[['Score']]
    # exctract tfidf features
    vectorizer = TfidfVectorizer(
        min_df=0.01, max_df=0.8, encoding='utf-8', use_idf=True, sublinear_tf=True, stop_words='english')
    X = vectorizer.fit_transform(df.Text)
    X = pd.DataFrame(X.toarray())
    # merge XY into a dataframe
    # drop the index column , for not confusing the concatenation
    new_df = pd.concat([X.reset_index(drop=True),Y.reset_index(drop=True)],axis=1,join="inner")
    # export
    new_df.to_csv('../dataset/tfidf_extracted.csv',index=False)
    

def extractWord2VEC():
    docs = [] 
    analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
    # transform the data
    for i, text in enumerate(df[['Text']].to_numpy()):
        # returns an array with one string , we get the string
        text = text[0]
        words = text.lower().split()
        tags = [i]
        docs.append(analyzedDocument(words, tags))
    # hyper parameters taken from paper , feed model
    model = Doc2Vec(docs,min_count=1,window=10,vector_size=100,sample = 0.0001,negative=5,workers=7,dm=1)
    # extract vectors
    X = []
    for i in range(0,len(model.dv)):
        X.append(model.dv[i])
    X = pd.DataFrame(X)
    Y = df[['Score']]
    # concat X and Y
    new_df =  pd.concat([X.reset_index(drop=True),Y.reset_index(drop=True)],axis=1,join="inner")
    # export
    new_df.to_csv('../dataset/doc2vec_extracted.csv',index=False)

extractTFIDF()
# extractWord2VEC()

# %%
