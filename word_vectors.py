import logging
import gensim
from gensim.models import KeyedVectors
from sklearn.decomposition import IncrementalPCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO)

# load trained model
# to do: correct file name of the model..
wv_model = gensim.models.KeyedVectors.load_word2vec_format('/Users/Lara/Desktop/Uni/Info_4/Masterarbeit/DATA/HMP/anno_corpus/corpus/models/genism-model')

#print(wv_model['anmuthig'])

# 100 most frequent words
print(wv_model.index_to_key[:100])

X = wv_model[wv_model.key_to_index]
print(len(X))
print(X[0])
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X[:1000,:])

plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
plt.show()


