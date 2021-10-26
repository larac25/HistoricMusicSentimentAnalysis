import logging
import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import altair as alt
alt.renderers.enable('altair_viewer')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
import numpy as np

training_path = '/Users/Lara/Desktop/Uni/Info_4/Masterarbeit/DATA/HMP/anno_corpus/corpus/label/data/train_1/'
test_path = '/Users/Lara/Desktop/Uni/Info_4/Masterarbeit/DATA/HMP/anno_corpus/corpus/label/data/test_1/'

final_train_df = os.path.join(training_path, 'train_df.csv')
final_test_df = os.path.join(test_path, 'test_df.csv')

if not os.path.exists(final_train_df):

    # merge train cvs files into one big dataframe
    train_list = []

    for train in os.listdir(training_path):
        tr = pd.read_csv(os.path.join(training_path, train), index_col=0)
        train_list.append(tr)

    # final training dataframe which includes all csv files from train files
    train_df = pd.concat(train_list)

    test_list = []

    for test in os.listdir(test_path):
        te = pd.read_csv(os.path.join(test_path, test), index_col=0)
        test_list.append(te)

    # final test dataframe which includes all csv files from test files
    test_df = pd.concat(test_list)

    # save final dataframes for training and testing
    with open(os.path.join(training_path, 'train_df.csv'), 'w') as out_train_df:
        train_df.to_csv(out_train_df)

    with open(os.path.join(test_path, 'test_df.csv'), 'w') as out_test_df:
        test_df.to_csv(out_test_df)

else:
    print('The final dataframes have already been created. Processing with label coding..')

    train_df = pd.read_csv(final_train_df, index_col=0)
    test_df = pd.read_csv(final_test_df)

viz = input('do you want to plot the visualisation? y / n')

if viz == 'y':
    # plot explorative analysis (show presence of different labels)
    # disable max_rows because our dataframe is too large
    alt.data_transformers.disable_max_rows()

    # the visualisation shows that the dataset is not balanced --> sentences without a label occur extremely often
    bars = alt.Chart(train_df).mark_bar(size=50).encode(
        x=alt.X("label"),
        y=alt.Y("count():Q", axis=alt.Axis(title='Number of sentences')),
        tooltip=[alt.Tooltip('count()', title='Number of sentences'), 'label'],
        color='label'

    ).properties(width=1000)

    text = bars.mark_text(
        align='center',
        baseline='bottom',
    ).encode(
        text='count()'
    )

    # set parameters for interactive info pop up
    (bars + text).interactive().properties(
        height=300,
        width=700,
        title="Number of sentences in each category",
    )

    bars.show()
    bars.save('data_viz.html')

if viz == 'n':
    pass

# label coding
train_df['label'] = train_df['label'].fillna(value=0)
# --> 0-9 (0 = nan, 1-9 = categories)
label_codes = {
    'anmuthig': 1,
    'bruetend': 2,
    'duester': 3,
    'ergreifend': 4,
    'feurig': 5,
    'leidenschaftlich': 6,
    'trotzig': 7,
    'trueb': 8,
    'wild': 9
}

# label mapping
train_df['label_code'] = train_df['label']
train_df = train_df.replace({'label_code': label_codes})


# Feature Engineering
# train / test splitting
x_train, x_test, y_train, y_test = train_test_split(train_df['text'], train_df['label_code'],
                                                    test_size=0.20, random_state=10)

# use TF-IDF Vectors as features
tfidf = TfidfVectorizer(encoding='utf-8', ngram_range=(1,2), min_df=5)
features_train = tfidf.fit_transform(x_train).toarray()
labels_train = y_train
print(features_train.shape)

features_test = tfidf.transform(x_test).toarray()
labels_test = y_test
print(features_test.shape)

# todo: Predictive Model

# todo: Evaluation

# todo: save model