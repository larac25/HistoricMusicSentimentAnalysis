import logging
import os
import pandas as pd
import gensim
from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle


logging.basicConfig(level=logging.INFO)

# load trained model

wv_model = gensim.models.KeyedVectors.load_word2vec_format('/Users/Lara/Desktop/Uni/Info_4/Masterarbeit/DATA/HMP'
                                                           '/anno_corpus/corpus/models/gensim-model')

'''
# get vectors for words from seed list
seed_vectors = list()
seed_vectors.append(wv_model.get_vector('anmuthig'))
seed_vectors.append(wv_model.get_vector('bruetend'))
seed_vectors.append(wv_model.get_vector('duester'))
seed_vectors.append(wv_model.get_vector('ergreifend'))
seed_vectors.append(wv_model.get_vector('feurig'))
seed_vectors.append(wv_model.get_vector('leidenschaftlich'))
seed_vectors.append(wv_model.get_vector('trotzig'))
seed_vectors.append(wv_model.get_vector('trueb'))
seed_vectors.append(wv_model.get_vector('wild'))

for vec in seed_vectors:
    sim_vecs = wv_model.similar_by_vector(vec, topn=10)
'''

# 100 most frequent words
print(wv_model.index_to_key[:100])


# new directory to store label-thesaurus
def outp_dir():
    outp_path = '/Users/Lara/Desktop/Uni/Info_4/Masterarbeit/DATA/HMP/anno_corpus/corpus/label/'
    if not os.path.exists(outp_path):
        os.makedirs(outp_path)

    return outp_path


switch = input('choose option to get most similar words: ms = most_similar; msc = most_similar_cosmul')


# get similar words for seed words of seed word list
seed_words = ['anmuthig', 'bruetend', 'duester', 'ergreifend', 'feurig', 'leidenschaftlich', 'trotzig', 'trueb', 'wild']
seeds = []  # store most similar words (for txt file)

# create new dataframe
new_df = pd.DataFrame()

for word in seed_words:
    if word in wv_model.index_to_key:

        # todo: add more options to gather similar words for seed words

        if switch == 'ms':

            # gensim.word2vec built in function
            similar = wv_model.most_similar(positive=word, topn=50)
            print(word, ':', similar)

            similar_str = []

            for sim in similar:
                seeds.append(sim[0])  # only store the key (the str)
                similar_str.append(sim[0])  # store words for excel export

            # add new columns to dataframe
            new_df[word] = similar_str

            # todo: nur Adverbien
            # todo: verschiedene Wortformen rausfiltern (brauche keine Varianten)

        elif switch == 'msc':

            # gensim.word2vec built in function
            similar = wv_model.most_similar_cosmul(positive=word, topn=50)
            print(word, ':', similar)

            similar_str = []

            for sim in similar:
                seeds.append(sim[0])  # only store the key (the str)
                similar_str.append(sim[0])  # store words for excel export

            # für neue Wörter aus der seed_words liste ebenfalls most similar Wörter suchen --> Anzahl Durchläufe fixen
            # oder andere Methode testen? Z.B. über Entfernung

    else:
        print('Could not find any similar words!')

# export dataframe to excel sheet
with pd.ExcelWriter('/Users/Lara/Desktop/Uni/Info_4/Masterarbeit/DATA/HMP/anno_corpus/corpus/label/label.xlsx') as writer:
    new_df.to_excel(writer, sheet_name='most_sim')

# export dataframe with pickle to use it in labeling.py
new_df.to_pickle('/Users/Lara/Desktop/Uni/Info_4/Masterarbeit/DATA/HMP/anno_corpus/corpus/label/label.pkl')

seed_words.extend(seeds)

seed_vectors = []
for w in seed_words:
    seed_vectors.append(wv_model[w])

# save output (keyed vectors and txt file with seed_words and similar words)
kv = KeyedVectors(vector_size=wv_model.vector_size)  # create new (empty) KeyedVectors object
kv.add_vectors(seed_words, seed_vectors)  # add vectors from above
# save new keyed vectors
fname = outp_dir() + 'label_' + str(switch)
kv.save_word2vec_format(fname)

# save txt file
txt_name = outp_dir() + str(switch) + '.txt'
with open(txt_name, 'w+') as fp:
    for sw in seed_words:
        fp.write(sw + '\n')


# visualize word vectors
most_sim_model = gensim.models.KeyedVectors.load_word2vec_format(fname)
vocab = list(most_sim_model.key_to_index)
X = most_sim_model[vocab]
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X[:1000, :])

plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
plt.show()
