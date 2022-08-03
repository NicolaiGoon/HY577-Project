# %%
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../dataset/Reviews.csv',
                 usecols=['Text', 'Score'], index_col=False)
# df = pd.read_csv('../dataset/tfidf_extracted.csv')
# df = pd.read_csv('../dataset/doc2vec_extracted.csv')




df.Score.hist(xlabelsize=5)
plt.show()

# Score 1,2 means negative
df.Score[df.Score < 3] = -1
# Score 4,5 is positive
df.Score[df.Score > 3] = 1
# Score 3 is neutral
df.Score[df.Score == 3] = 0

df.Score.hist()
plt.show()
# %%
