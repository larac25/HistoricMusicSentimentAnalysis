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
    fname = outp_path + 'gensim-model'
    sentences = ModelInput('/Users/Lara/Desktop/Uni/Info_4/Masterarbeit/DATA/HMP/anno_corpus/corpus/prep_files')
    model = gensim.models.Word2Vec(sentences, min_count=5, vector_size=100, sg=0)

    wv = model.wv
    # retrieving vocabulary
    for index, word in enumerate(wv.index_to_key):
        if index == 50:
            break
        print(f"word #{index}/{len(wv.index_to_key)} is {word}")

    # model params:
    # min_count = ignores all words with total frequency lower than this
    # vector_size = dimensionality of the word vectors
    # sg = training algorithm (1=skipgram, 0=cbow)

    # saving the model
    model.wv.save_word2vec_format(fname)

    logging.info('trained model saved under outp_path')


if __name__ == '__main__':
    main()
