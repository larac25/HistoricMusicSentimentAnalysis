import logging
import os
import os.path
import shutil
import random
import pandas as pd
import csv


logging.basicConfig(level=logging.INFO)


# new directory to store labelled data
def outp_dir():
    outp_path = '/Users/Lara/Desktop/Uni/Info_4/Masterarbeit/DATA/HMP/anno_corpus/corpus/classifier/'
    if not os.path.exists(outp_path):
        os.makedirs(outp_path)

    return outp_path


# split data because of RAM Issues
def split_data():
    # extract 50% of the new corpus for each labeling process
    # new folders: word2vec & fasttext
    input_folder = '/Users/Lara/Desktop/Uni/Info_4/Masterarbeit/DATA/HMP/anno_corpus/corpus/classifier/'
    split_files = random.sample(os.listdir(input_folder), 150) # ATTENTION changed number
    wv_folder = '/Users/Lara/Desktop/Uni/Info_4/Masterarbeit/DATA/HMP/anno_corpus/corpus/classifier/word2vec/'
    ft_folder = '/Users/Lara/Desktop/Uni/Info_4/Masterarbeit/DATA/HMP/anno_corpus/corpus/classifier/fasttext/'

    # check if wv_folder already exists
    if not os.path.exists(wv_folder):
        os.makedirs(wv_folder)

        # move randomly selected files to new folder
        for wv_files in split_files:
            old_path = os.path.join(input_folder, wv_files)
            shutil.move(old_path, wv_folder)

    # check if ft_folder already exists
    if not os.path.exists(ft_folder):
        os.makedirs(ft_folder)

        # these are the files left from the splitting
        ft_files = os.listdir(input_folder)

        # move all csv files to new test folder
        for test in filter(lambda test: test.endswith('.csv'), ft_files):  # only consider csv
            # --> otherwise the wv_folder is moved, too
            old_p = os.path.join(input_folder, test)
            shutil.move(old_p, ft_folder)


# string-matching-algorithm to label sentences with seed word categories
def label_data():

    label_inpput = input('which embeddings for labeling? wv = word2vec; ft = fasttext')

    if label_inpput == 'wv':

        # load word2vec data and word2vec generated emotion lexicon
        data = '/Users/Lara/Desktop/Uni/Info_4/Masterarbeit/DATA/HMP/anno_corpus/corpus/classifier/word2vec/'
        label = pd.read_pickle('/Users/Lara/Desktop/Uni/Info_4/Masterarbeit/DATA/HMP/anno_corpus/corpus/label/label_wv.pkl')

        # sentiment codes from Emotionlexicon with Word2Vec Embeddings
        sentiment_codes = {
            'anmuthig': 'p',
            'bruetend': 'n',
            'duester': 'n',
            'ergreifend': 'p',
            'feurig': 'p',
            'leidenschaftlich': 'p',
            'trotzig': 'n',
            'trueb': 'n',
            'wild': 'n'
        }

    elif label_inpput == 'ft':

        # load fasttext data and fasttext generated emotion lexicon
        data = '/Users/Lara/Desktop/Uni/Info_4/Masterarbeit/DATA/HMP/anno_corpus/corpus/classifier/fasttext/'
        label = pd.read_pickle('/Users/Lara/Desktop/Uni/Info_4/Masterarbeit/DATA/HMP/anno_corpus/corpus/label/label_ft.pkl')

        # sentiment codes from Emotionslexicon with FastText Embeddings
        sentiment_codes = {
            'anmuthig': 'p',
            'bruetend': 'n',
            'duester': 'p',
            'ergreifend': 'n',
            'feurig': 'p',
            'leidenschaftlich': 'p',
            'trotzig': 'n',
            'trueb': 'n',
            'wild': 'n'
        }

    # read csv files as stream
    for root_name, dir_n, csv_files in os.walk(data):
        print(f'Found directory: {root_name}')
        csv_num = len(csv_files)
        print('found', csv_num, 'csv_files')

        for f in csv_files:
            # open csv-file
            with open(os.path.join(root_name, f), 'r') as in_csv:
                # read csv as dataframe
                f_df = pd.read_csv(in_csv)
                f_df['label'] = None

                # check if any word from emotion lexicon is present in each row
                # if yes: label with specific seed word(s)
                # if not: no label

                for index, row in f_df.iterrows():

                    for rowIndex, word in label.iterrows():
                        for columnIndex, value in word.items():

                            if row.str.contains(value).all():
                                # update column 'label' with seed word category
                                f_df.loc[index, 'label'] = columnIndex

                # label mapping
                f_df['sentiment'] = f_df['label']
                f_df = f_df.replace({'sentiment': sentiment_codes})

                with open(os.path.join(root_name, f), 'w') as labelled_file:
                    f_df.to_csv(labelled_file)


# read data as stream and convert 50% of it to csv (only 50% because of RAM issues)
raw_data = '/Users/Lara/Desktop/Uni/Info_4/Masterarbeit/DATA/HMP/anno_corpus/corpus/prep_files/'
csv_path = outp_dir()

# check if data already has been created --> no file converting and splitting necessary
if len(os.listdir(csv_path)) == 0:

    for root, dirnames, files in os.walk(raw_data):
        print(f'Found directory: {root}')
        file_number = len(files)
        print('found', file_number, 'files')

        files = random.sample(os.listdir(raw_data), 300) # ATTENTION: changed number

        for filename in files:
            # create output directory and filename for processed files
            outp_file = csv_path + str(os.path.splitext(filename)[0]) + ".csv"

            # strip lines and write one sentence per row
            with open(os.path.join(root, filename), 'r') as in_file:
                stripped = (line.strip() for line in in_file)
                lines = (line.split(",") for line in stripped if line)
                with open(outp_file, 'w') as out_file:
                    writer = csv.writer(out_file)
                    writer.writerow(('text', ))
                    writer.writerows(lines)

    # call function to split data
    split_data()

else:
    print('data has already been created')

# write labelled data as output
label_data()

