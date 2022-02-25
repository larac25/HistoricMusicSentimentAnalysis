import os
import random
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import altair as alt
sns.set_style('whitegrid')
alt.renderers.enable('altair_viewer')

#Importieren der Module zur Erstellung des Modells
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV

#Importieren der Klassifizierungs Algorithmen
from sklearn.svm import LinearSVC

clf_input = input('which labelled data? wv = labelled data with word2vec embeddings; ft = labelled data with fasttext embeddings')

if clf_input == 'wv':

    training_path = '/Users/Lara/Desktop/Uni/Info_4/Masterarbeit/DATA/HMP/anno_corpus/corpus/classifier/word2vec/'

elif clf_input == 'ft':

    training_path = '/Users/Lara/Desktop/Uni/Info_4/Masterarbeit/DATA/HMP/anno_corpus/corpus/classifier/fasttext/'

final_train_df = os.path.join(training_path, 'train_df.csv')

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

        # save final dataframe for training
        with open(os.path.join(training_path, 'train_df.csv'), 'w') as out_train_df:
            train_df.to_csv(out_train_df)

    if data_size == 'y':

        # randomly sample x files due to memory issues
        train_files = random.sample(os.listdir(training_path), 150)
        # merge train csv files into one big dataframe
        less_files_list = []

        for less_files in train_files:
            less_tr = pd.read_csv(os.path.join(training_path, less_files), index_col=0)
            less_files_list.append(less_tr)

        # final training dataframe which includes all csv files from train files
        half_train_df = pd.concat(less_files_list)

        # save final dataframes for training and testing
        with open(os.path.join(training_path, 'train_df.csv'), 'w') as out_half_df:
            half_train_df.to_csv(out_half_df)

        train_df = pd.read_csv(final_train_df, index_col=0)


else:
    print('The final dataframes have already been created. Processing with label coding..')

    train_df = pd.read_csv(final_train_df, index_col=0)


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

if viz == 'n':
    pass

cl_input = input('sentiment or binary classifier? sentiment/binary')

# save data, features and tfidf-object
pickle_path = '/Users/Lara/Desktop/Uni/Info_4/Masterarbeit/DATA/HMP/anno_corpus/corpus/classifier/pickle'

if not os.path.exists(pickle_path):
    os.makedirs(pickle_path)

if cl_input == 'sentiment':
    # label coding
    # --> -1 to 1 (-1 = negative,  0 = no label, 1 = positive)
    train_df.drop(columns='label', inplace=True, axis=1)
    train_df['sentiment'] = train_df['sentiment'].fillna(value=0)
    label_codes = {
        'p': 1,
        'n': -1,
    }

    # label mapping
    train_df['label_code'] = train_df['sentiment']
    train_df = train_df.replace({'label_code': label_codes})

    # Pipeline fasst Prozess zusammen: Vektorisieren --> Transformieren --> Klassifizieren
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', LinearSVC(multi_class='ovr'))])

    # Parameter für Grid Search festlegen
    tuned_parameters = {
        'vect__ngram_range': [(1, 2), (1, 3), (1, 4)],
        'tfidf__use_idf': (True, False),
        'clf__tol': [1, 1e-1, 1e-2, 1e-3]
    }

    # Datenset splitten
    x_train, x_test, y_train, y_test = train_test_split(train_df['text'], train_df['label_code'],
                                                        test_size=0.2, shuffle=True, random_state=10)

    clf = GridSearchCV(text_clf, tuned_parameters, cv=10, n_jobs=2)

    # Modell trainieren
    clf.fit(x_train, y_train)

    # classification report für Analyse mit SVM
    print(classification_report(y_test, clf.predict(x_test), digits=4))

    # Accuracy für Analyse mit SVM
    print(accuracy_score(y_test, clf.predict(x_test)))

    # Ausgabe der besten Parameter
    print("Best Score: ", clf.best_score_)
    print("Best Params: ", clf.best_params_)

    # confusion matrix
    aux_df = train_df[['sentiment', 'label_code']].drop_duplicates().sort_values('label_code')
    conf_matrix = confusion_matrix(y_test, clf.predict(x_test), normalize='true')
    plt.figure(figsize=(12.8, 6))
    sns.heatmap(conf_matrix,
                annot=True,
                xticklabels=aux_df['sentiment'].values,
                yticklabels=aux_df['sentiment'].values,
                cmap="Blues")
    plt.ylabel('Predicted')
    plt.xlabel('Actual')
    plt.title('Confusion matrix')
    plt.show()

    # save model
    with open(os.path.join(pickle_path, 'multi_clf.pickle'), 'wb') as output:
        pickle.dump(clf, output)


if cl_input == 'binary':

    # label coding
    # (0 = no label, 1 = any label present)
    train_df.drop(columns='sentiment', inplace=True, axis=1)
    train_df['label'] = train_df['label'].fillna(value=0)
    train_df['label'][~train_df['label'].isin([0])] = 'labelled'
    label_codes = {
        'labelled': 1,
    }

    # label mapping
    train_df['label_code'] = train_df['label']
    train_df = train_df.replace({'label_code': label_codes})

    # Pipeline fasst Prozess zusammen: Vektorisieren --> Transformieren --> Klassifizieren
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', LinearSVC(class_weight={1: 10}))])

    # Parameter für Grid Search festlegen
    tuned_parameters = {
        'vect__ngram_range': [(1, 2), (1, 3), (1, 4)],
        'tfidf__use_idf': (True, False),
        'clf__tol': [1, 1e-1, 1e-2, 1e-3]
    }

    # Datenset splitten
    x_train, x_test, y_train, y_test = train_test_split(train_df['text'], train_df['label_code'],
                                                        test_size=0.2, shuffle=True, random_state=10)

    clf = GridSearchCV(text_clf, tuned_parameters, cv=10, n_jobs=2)

    # Modell trainieren
    clf.fit(x_train, y_train)

    # classification report für Analyse mit SVM
    print(classification_report(y_test, clf.predict(x_test), digits=4))

    # Accuracy für Analyse mit SVM
    print(accuracy_score(y_test, clf.predict(x_test)))

    # Ausgabe der besten Parameter
    print("Best Score: ", clf.best_score_)
    print("Best Params: ", clf.best_params_)

    # confusion matrix
    aux_df = train_df[['label', 'label_code']].drop_duplicates().sort_values('label_code')
    conf_matrix = confusion_matrix(y_test, clf.predict(x_test), normalize='true')
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
    with open(os.path.join(pickle_path, 'binary_clf.pickle'), 'wb') as output:
        pickle.dump(clf, output)
