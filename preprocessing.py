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
from gensim.models import Phrases


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

            # call functions for preprocessing
            document = umlaute(document)
            document = prep(document)
            out_dir = outp_dir()
            # break document into utf8 tokens
            #yield gensim.utils.tokenize(document, lower=True, errors='ignore')

            # create filename for preprocessed original file and write processed data
            prep_file = out_dir + str(os.path.splitext(fname)[0]) + "_prep.txt"
            with open(prep_file, 'w+') as output:
                for sentence in document:
                    output.write(sentence + '\n')
                    
        # to do: understand this part!!!
        sentences = PathLineSentences(out_dir)
        phrases = Phrases(sentences, min_count=5, threshold=10)
        # print(list(phrases[sentences]))

        filenames = os.listdir(out_dir)
        for file in filenames:
            path = os.path.join(out_dir, file)
            with open(path, "r+") as f:
                tokenized_sentences = phrases[LineSentence(path)]
                f.seek(0)
                for s in tokenized_sentences:
                    f.write('{}\n'.format(' '.join(s)))
                f.truncate()

            # to do: one sentence per line


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
    document = document.replace('ſ', 's')
    return document


def outp_dir():
    outp_path = '/Users/Lara/Desktop/Uni/Info_4/Masterarbeit/DATA/HMP/anno_corpus/corpus/prep_files/'
    if not os.path.exists(outp_path):
        os.makedirs(outp_path)

    return outp_path


logging.info('processed all files')


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
