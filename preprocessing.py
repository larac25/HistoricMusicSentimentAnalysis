import logging
import argparse
import os
import gensim
from gensim.corpora import textcorpus
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from gensim.parsing.preprocessing import strip_punctuation
from gensim.parsing.preprocessing import strip_non_alphanum
from gensim.models.word2vec import LineSentence
from gensim.models.word2vec import PathLineSentences


logging.basicConfig(level=logging.INFO)


def main():

    top_directory = '/Users/Lara/Desktop/Uni/Info_4/Masterarbeit/DATA/HMP/anno_corpus/corpus/input_txt'
    # stream files from directory
    for root, dirnames, files in os.walk(top_directory):
        #print(f'Found directory: {root}')
        file_number = len(files)
        print('found', file_number, 'files')

        for fname in filter(lambda fname: fname.endswith('.txt'), files):
            # read each document as one big string
            document = open(os.path.join(root, fname)).read()

            prep(document)
            umlaute(document)
            # break document into utf8 tokens
            #yield gensim.utils.tokenize(document, lower=True, errors='ignore')

            return document


def prep(text):
    document = text
    custom_filters = [lambda x: x.lower(), remove_stopwords, strip_multiple_whitespaces, strip_punctuation,
                      strip_non_alphanum]
    document = preprocess_string(document, custom_filters)
    return document


def umlaute(text):
    document = text
    document = document.replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue')
    document = document.replace('Ä', 'Ae').replace('Ö', 'Oe').replace('Ü', 'Ue')
    document = document.replace('ß', 'ss')
    return document


logging.info('collected all files in directory')


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
