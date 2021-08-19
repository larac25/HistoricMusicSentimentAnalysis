import logging
import argparse
import os
import gensim
from gensim.corpora import textcorpus
from gensim.parsing.preprocessing import preprocess_string
from gensim.models.word2vec import LineSentence
from gensim.models.word2vec import PathLineSentences


logging.basicConfig(level=logging.INFO)


def main():

    top_directory = '/Users/Lara/Desktop/Uni/Info_4/Masterarbeit/DATA/HMP/anno_corpus/corpus/input_txt'
    # stream files from directory
    for root, dirnames, files in os.walk(top_directory):
        print(f'Found directory: {root}')
        file_number = len(files)
        print('found', file_number, 'files')

        for fname in filter(lambda fname: fname.endswith('.txt'), files):
            # read each document as one big string
            document = open(os.path.join(root, fname)).read()
            # to do: break document into utf8 tokens
            return document
            #yield gensim.utils.tokenize(document, lower=True, errors='ignore')


logging.info('collected all files in directory')


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
