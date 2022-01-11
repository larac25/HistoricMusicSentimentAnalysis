import logging
import os
import pandas as pd
import gensim
from gensim.models import FastText
from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO)


# new directory to store label-thesaurus
def outp_dir():
    outp_path = '/Users/Lara/Desktop/Uni/Info_4/Masterarbeit/DATA/HMP/anno_corpus/corpus/label/'
    if not os.path.exists(outp_path):
        os.makedirs(outp_path)

    return outp_path


switch = input('choose option to get most similar words: wv = word2vec model; ft = fasttext model')


# get similar words for seed words of seed word list
seed_words = ['anmuthig', 'bruetend', 'duester', 'ergreifend', 'feurig', 'leidenschaftlich', 'trotzig', 'trueb', 'wild']
seeds = []  # store most similar words (for txt file)

# create new dataframe
new_df = pd.DataFrame()

# create new output directory
outp_dir()

if switch == 'wv':

    # load word2vec model
    wv_model = gensim.models.KeyedVectors.load_word2vec_format('/Users/Lara/Desktop/Uni/Info_4/Masterarbeit/DATA/HMP'
                                                               '/anno_corpus/corpus/models/word2vec')

    for word in seed_words:

        if word in wv_model.index_to_key:

            # gensim.word2vec built in function
            similar = wv_model.most_similar(positive=word, topn=25)
            print(word, ':', similar)

            similar_str = []

            for sim in similar:
                seeds.append(sim[0])  # only store the key (the str)
                similar_str.append(sim[0])  # store words for excel export

            # add new columns to dataframe
            new_df[word] = similar_str

        else:
            print('Could not find any similar words!')

    # export dataframe to excel sheet
    with pd.ExcelWriter(
            '/Users/Lara/Desktop/Uni/Info_4/Masterarbeit/DATA/HMP/anno_corpus/corpus/label/label_wv.xlsx') as writer:
        new_df.to_excel(writer, sheet_name='most_sim')
        writer.save()

    # export dataframe with pickle to use it in labeling.py
    new_df.to_pickle('/Users/Lara/Desktop/Uni/Info_4/Masterarbeit/DATA/HMP/anno_corpus/corpus/label/label_wv.pkl')

    seed_words.extend(seeds)

    seed_vectors = []
    for w in seed_words:
        seed_vectors.append(wv_model[w])

    # save output (keyed vectors and txt file with seed_words and similar words) from word2vec embeddings
    kv_ft = KeyedVectors(vector_size=wv_model.vector_size)  # create new (empty) KeyedVectors object
    kv_ft.add_vectors(seed_words, seed_vectors)  # add vectors from above
    # save new keyed vectors
    fname = outp_dir() + 'label_' + str(switch)
    kv_ft.save_word2vec_format(fname)

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


elif switch == 'ft':

    # load fasttext model
    ft_model = gensim.models.KeyedVectors.load(
        '/Users/Lara/Desktop/Uni/Info_4/Masterarbeit/DATA/HMP/anno_corpus/corpus/models/fasttext')

    for word in seed_words:

        if word in ft_model.index_to_key:

            similar = ft_model.most_similar(positive=word, topn=25)
            print(word, ':', similar)

            similar_str = []

            for sim in similar:
                seeds.append(sim[0])  # only store the key (the str)
                similar_str.append(sim[0])  # store words for excel export

            # add new columns to dataframe
            new_df[word] = similar_str

        else:
            print('Could not find any similar words!')

    # export dataframe to excel sheet
    with pd.ExcelWriter(
            '/Users/Lara/Desktop/Uni/Info_4/Masterarbeit/DATA/HMP/anno_corpus/corpus/label/label_ft.xlsx') as writer:
        new_df.to_excel(writer, sheet_name='most_sim')
        writer.save()

    # export dataframe with pickle to use it in labeling.py
    new_df.to_pickle('/Users/Lara/Desktop/Uni/Info_4/Masterarbeit/DATA/HMP/anno_corpus/corpus/label/label_ft.pkl')

    seed_words.extend(seeds)

    seed_vectors = []
    for w in seed_words:
        seed_vectors.append(ft_model[w])

    # save output (keyed vectors and txt file with seed_words and similar words) from fasttext embeddings
    kv_ft = KeyedVectors(vector_size=ft_model.vector_size)  # create new (empty) KeyedVectors object
    kv_ft.add_vectors(seed_words, seed_vectors)  # add vectors from above
    # save new keyed vectors
    fname = outp_dir() + 'label_' + str(switch)
    kv_ft.save_word2vec_format(fname)

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
