import logging
import os
import gensim

logging.basicConfig(level=logging.INFO)


# class for input files (creates an iterator for data streaming)
class ModelInput(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()


# new directory to store models
def outp_dir():
    outp_path = '/Users/Lara/Desktop/Uni/Info_4/Masterarbeit/DATA/HMP/anno_corpus/corpus/models/'
    if not os.path.exists(outp_path):
        os.makedirs(outp_path)

    return outp_path


def main():
    outp_path = outp_dir()
    fname_word2vec = outp_path + 'word2vec'
    fname_fasttext = outp_path + 'fasttext'
    sentences = ModelInput('/Users/Lara/Desktop/Uni/Info_4/Masterarbeit/DATA/HMP/anno_corpus/corpus/prep_files')

    # ----------------------Word2Vec---------------------
    w2v_model = gensim.models.Word2Vec(sentences, min_count=5, vector_size=100, sg=0)
    wv_vocab = w2v_model.wv
    # retrieving vocabulary
    for index, word in enumerate(wv_vocab.index_to_key):
        if index == 50:
            break
        print(f"word #{index}/{len(wv_vocab.index_to_key)} is {word}")

    # word2vec model params:
    # min_count = ignores all words with total frequency lower than this
    # vector_size = dimensionality of the word vectors
    # sg = training algorithm (1=skipgram, 0=cbow)

    # saving the model
    w2v_model.wv.save_word2vec_format(fname_word2vec)

    # ---------------------FastText----------------------
    ft_model = gensim.models.FastText(sentences, min_count=5, vector_size=100)
    ft_vocab = ft_model.wv
    # retrieving vocabulary
    for index, word in enumerate(ft_vocab.index_to_key):
        if index == 50:
            break
        print(f"word #{index}/{len(ft_vocab.index_to_key)} is {word}")

    # fasttext model params:
    # min_count = ignores all words with total frequency lower than this
    # vector_size = dimensionality of the word vectors

    # saving the model
    ft_model.wv.save(fname_fasttext)

    logging.info('trained models saved under outp_path')


if __name__ == '__main__':
    main()
