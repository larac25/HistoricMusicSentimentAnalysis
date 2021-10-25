import logging
import os
import os.path
import shutil
import random
import pandas as pd
import pickle
import csv


logging.basicConfig(level=logging.INFO)


# new directory to store labelled data
def outp_dir():
    outp_path = '/Users/Lara/Desktop/Uni/Info_4/Masterarbeit/DATA/HMP/anno_corpus/corpus/label/data/'
    if not os.path.exists(outp_path):
        os.makedirs(outp_path)

    return outp_path


# split data into training and validation set
def split_data():
    # extract training data
    # leave some of the data for testing
    # new folders: train & test
    input_folder = '/Users/Lara/Desktop/Uni/Info_4/Masterarbeit/DATA/HMP/anno_corpus/corpus/label/data/'
    train_files = random.sample(os.listdir(input_folder), 8414)  # randomly sample approx. 80% of all files for training
    train_folder = '/Users/Lara/Desktop/Uni/Info_4/Masterarbeit/DATA/HMP/anno_corpus/corpus/label/data/train/'
    test_folder = '/Users/Lara/Desktop/Uni/Info_4/Masterarbeit/DATA/HMP/anno_corpus/corpus/label/data/test/'

    # check if 'train' folder already exists
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)

        # move randomly selected files to new train folder
        for train in train_files:
            old_path = os.path.join(input_folder, train)
            shutil.move(old_path, train_folder)

    # check if 'test' folder already exists
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

        # these are the files left from the splitting (20% from the whole set)
        test_files = os.listdir(input_folder)

        # move all csv files to new test folder
        for test in filter(lambda test: test.endswith('.csv'), test_files):  # only consider csv
            # --> otherwise the 'train' folder is moved, too
            old_p = os.path.join(input_folder, test)
            shutil.move(old_p, test_folder)


# string-matching-algorithm to label sentences with seed word categories
def label_data():
    # load training data and emotion lexicon
    data = '/Users/Lara/Desktop/Uni/Info_4/Masterarbeit/DATA/HMP/anno_corpus/corpus/label/data/train/'
    label = pd.read_pickle('/Users/Lara/Desktop/Uni/Info_4/Masterarbeit/DATA/HMP/anno_corpus/corpus/label/label.pkl')

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

                with open(os.path.join(root_name, f), 'w') as labelled_file:
                    f_df.to_csv(labelled_file)


# read data as stream and convert to csv
raw_data = '/Users/Lara/Desktop/Uni/Info_4/Masterarbeit/DATA/HMP/anno_corpus/corpus/prep_files/'
csv_path = outp_dir()

# check if data already has been created --> no file converting and splitting necessary
if len(os.listdir(csv_path)) == 0:

    for root, dirnames, files in os.walk(raw_data):
        print(f'Found directory: {root}')
        file_number = len(files)
        print('found', file_number, 'files')

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

