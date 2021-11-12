import logging
import os
import random

import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
import numpy as np
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import altair as alt
sns.set_style('whitegrid')
alt.renderers.enable('altair_viewer')


training_path = '/Users/Lara/Desktop/Uni/Info_4/Masterarbeit/DATA/HMP/anno_corpus/corpus/label/data/train_1/'
test_path = '/Users/Lara/Desktop/Uni/Info_4/Masterarbeit/DATA/HMP/anno_corpus/corpus/label/data/test_1/'

final_train_df = os.path.join(training_path, 'train_df.csv')
final_test_df = os.path.join(test_path, 'test_df.csv')

if not os.path.exists(final_train_df):

    data_size = input('use less of the data? (to prevent memory issues) y / n')

    if data_size == 'n':

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

    if data_size == 'y':

        # randomly sample half of the data
        train_files = random.sample(os.listdir(training_path), 150)
        # merge train cvs files into one big dataframe
        half_train_list = []

        for half_train in train_files:
            half_tr = pd.read_csv(os.path.join(training_path, half_train), index_col=0)
            half_train_list.append(half_tr)

        # final training dataframe which includes all csv files from train files
        half_train_df = pd.concat(half_train_list)

        # save final dataframes for training and testing
        with open(os.path.join(training_path, 'train_df.csv'), 'w') as out_half_df:
            half_train_df.to_csv(out_half_df)

        train_df = pd.read_csv(final_train_df, index_col=0)


else:
    print('The final dataframes have already been created. Processing with label coding..')

    train_df = pd.read_csv(final_train_df, index_col=0)
#    test_df = pd.read_csv(final_test_df)


row_count = train_df.count()
word_count = [len(x.split()) for x in train_df['text'].tolist()]
print(row_count)
print(word_count)


viz = input('do you want to plot the visualisation? y / n')

if viz == 'y':
    # plot explorative analysis (show presence of different labels)
    # disable max_rows because our dataframe is too large
    alt.data_transformers.disable_max_rows()

    # the visualisation shows that the dataset is not balanced --> sentences without a label occur extremely often
    # sentences without a label should be considered when training the classifier
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


# generator to stream training data in tfidf-vectorizer
def ChunkIterator(filename):
    for chunk in filename.iteritems():
        yield chunk


# use TF-IDF Vectors as features
tfidf = TfidfVectorizer(encoding='utf-8', ngram_range=(1, 2), min_df=5)
features_train = tfidf.fit_transform(list(zip(*ChunkIterator(x_train)))[1]).toarray()
labels_train = y_train
print('features_train:', features_train.shape)

features_test = tfidf.transform(list(zip(*ChunkIterator(x_test)))[1]).toarray()
labels_test = y_test
print('features_test:', features_test.shape)

# save data, features and tfidf-object
pickle_path = '/Users/Lara/Desktop/Uni/Info_4/Masterarbeit/DATA/HMP/anno_corpus/corpus/label/data/pickle'

if not os.path.exists(pickle_path):
    os.makedirs(pickle_path)

'''
# X_train
with open(os.path.join(pickle_path, 'x_train.pickle'), 'wb') as output:
    pickle.dump(x_train, output)

# X_test
with open(os.path.join(pickle_path, 'x_test.pickle'), 'wb') as output:
    pickle.dump(x_test, output)

# y_train
with open(os.path.join(pickle_path, 'y_train.pickle'), 'wb') as output:
    pickle.dump(y_train, output)

# y_test
with open(os.path.join(pickle_path, 'y_test.pickle'), 'wb') as output:
    pickle.dump(y_test, output)

# df
with open(os.path.join(pickle_path, 'df.pickle'), 'wb') as output:
    pickle.dump(train_df, output)

# features_train
with open(os.path.join(pickle_path, 'features_train.pickle'), 'wb') as output:
    pickle.dump(features_train, output)

# labels_train
with open(os.path.join(pickle_path, 'labels_train.pickle'), 'wb') as output:
    pickle.dump(labels_train, output)

# features_test
with open(os.path.join(pickle_path, 'features_test.pickle'), 'wb') as output:
    pickle.dump(features_test, output)

# labels_test
with open(os.path.join(pickle_path, 'labels_test.pickle'), 'wb') as output:
    pickle.dump(labels_test, output)

# TF-IDF object
with open(os.path.join(pickle_path, 'tfidf.pickle'), 'wb') as output:
    pickle.dump(tfidf, output)
    
'''

# use the Chi squared test in order to see what unigrams and bigrams are most correlated with each category
# not working --> SIGKILL 9

for Product, label_id in sorted(label_codes.items()):
    features_chi2 = chi2(features_train, labels_train == label_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    print("# '{}' label:".format(Product))
    print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-5:])))
    print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-2:])))
    print("")

# Predictive Model (SVM)
clf = svm.LinearSVC()

clf.fit(features_train, labels_train)

# Evaluation

# accuracy score
print(accuracy_score(labels_test, clf.predict(features_test)))

# classification report
print(classification_report(labels_test, clf.predict(features_test)))

# confusion matrix
aux_df = train_df[['label', 'label_code']].drop_duplicates().sort_values('label_code')
conf_matrix = confusion_matrix(labels_test, clf.predict(features_test))
plt.figure(figsize=(12.8, 6))
sns.heatmap(conf_matrix,
            annot=True,
            xticklabels=aux_df['label'].values,
            yticklabels=aux_df['label'].values,
            cmap="Blues")
plt.ylabel('Predicted')
plt.xlabel('Actual')
plt.title('Confusion matrix')
plt.show()

# save model
with open(os.path.join(pickle_path, 'clf.pickle'), 'wb') as output:
    pickle.dump(clf, output)
