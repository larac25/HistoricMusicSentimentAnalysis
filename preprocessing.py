import logging
import os
import nltk
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from gensim.parsing.preprocessing import strip_punctuation
from gensim.parsing.preprocessing import strip_non_alphanum
from gensim.parsing.preprocessing import strip_short
from gensim.models.word2vec import LineSentence
from gensim.models.word2vec import PathLineSentences
from gensim.models import Phrases


logging.basicConfig(level=logging.INFO)


def main():

    top_directory = '/Users/Lara/Desktop/Uni/Info_4/Masterarbeit/DATA/HMP/anno_corpus/corpus/input_txt'
    # stream files from directory
    for root, dirnames, files in os.walk(top_directory):
        # print(f'Found directory: {root}')
        file_number = len(files)
        print('found', file_number, 'files')

        for fname in filter(lambda fname: fname.endswith('.txt'), files):
            # read each document as one big string
            document = open(os.path.join(root, fname)).read()

            # call function to create new output directory
            out_dir = outp_dir()

            # call function for preprocessing
            document = get_sentences(document)

            # create filename for preprocessed original file and write processed data
            prep_file = out_dir + str(os.path.splitext(fname)[0]) + "_prep.txt"
            with open(prep_file, 'w+', encoding='utf-8') as output:
                for sents in document:
                    output.write(str(sents) + '\n')

        # handle line splitting -> one sentence per line
        # handle n-grams
        sentences = PathLineSentences(out_dir)
        phrases = Phrases(sentences, min_count=5, threshold=10)

        filenames = os.listdir(out_dir)
        for file in filenames:
            path = os.path.join(out_dir, file)
            with open(path, "r+") as f:
                tokenized_sentences = phrases[LineSentence(path)]
                f.seek(0)
                for s in tokenized_sentences:
                    f.write('{}\n'.format(' '.join(s)))
                f.truncate()


def get_sentences(text):
    document = text
    # remove any newlines for sentence detection
    document = document.replace('¬\n', '').strip()
    document = document.replace('-\n', '')
    document = document.replace('\n', ' ')
    document = nltk.sent_tokenize(document, language='german')  # tokenization

    stop_words = stopwords.words('german')
    processed = []

    for sentences in document:
        sentences = umlaute(sentences)
        sentences = prep(sentences)

        # remove german stopwords
        sentences = [x for x in sentences if x not in stop_words]
        sentences = ' '.join(sentences)

        if len(sentences) > 1:
            processed.append(sentences)

    return processed


def prep(text):
    document = text
    # convert to lower, remove multiple whitespaces, punctuation, anything non-alphanumeric, short words
    custom_filters = [lambda x: x.lower(), strip_multiple_whitespaces, strip_punctuation,
                      strip_non_alphanum, strip_short]
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
