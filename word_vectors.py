import logging
import gensim
from gensim.models import KeyedVectors
from sklearn.decomposition import IncrementalPCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from gensim.scripts import word2vec2tensor


logging.basicConfig(level=logging.INFO)

# load trained model
# to do: correct file name of the model..
wv_model = gensim.models.KeyedVectors.load_word2vec_format('/Users/Lara/Desktop/Uni/Info_4/Masterarbeit/DATA/HMP/anno_corpus/corpus/models/genism-model')

'''
# get vectors for words from seed list
seed_vectors = list()
seed_vectors.append(wv_model.get_vector('anmuthig'))
seed_vectors.append(wv_model.get_vector('ergreifend'))
seed_vectors.append(wv_model.get_vector('feurig'))
seed_vectors.append(wv_model.get_vector('leidenschaftlich'))
seed_vectors.append(wv_model.get_vector('trotzig'))
seed_vectors.append(wv_model.get_vector('wild'))

for vec in seed_vectors:
    sim_vecs = wv_model.similar_by_vector(vec, topn=10)
'''


# 100 most frequent words
print(wv_model.index_to_key[:100])

# similar words for seed words of seed word list
seed_words = ['anmuthig', 'brütend', 'düster', 'ergreifend', 'feurig', 'leidenschaftlich', 'trotzig', 'trüb', 'wild']
seeds = []  # store most similar words (for tensorflow viz)

for word in seed_words:
    if word in wv_model.index_to_key:
        similar = wv_model.most_similar(positive=word, topn=10)
        print(word, ':', similar)

        for sim in similar:
            seeds.append(sim[0])  # only store the key (the str)

    else:
        print('Could not find any similar words!')

seed_words.extend(seeds)
seed_vectors = []
for w in seed_words:
    seed_vectors.append(wv_model[word])

kv = KeyedVectors(vector_size=wv_model.vector_size)  # create new (empty) KeyedVectors object
kv.add_vectors(seed_words, seed_vectors)  # add vectors from above
kv.save_word2vec_format(fname='/Users/Lara/Desktop/Uni/Info_4/Masterarbeit/DATA/HMP/anno_corpus/corpus/models/seed_vectors')

# visualize word vectors
'''
vocab = list(wv_model.key_to_index)
X = wv_model[vocab]
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X[:1000,:])

plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
plt.show()
'''