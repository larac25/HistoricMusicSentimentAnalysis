import logging
import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm

training_path = ''  # todo: add real path
test_path = ''  # todo: add real path

training_df = pd.read_csv(training_path)

# merge train cvs files into one big dataframe

# todo: label coding
# --> 0-9 (0-8 = categories, 9 = no label)

# todo: plot explorative analysis (show presence of different labels)

# todo: Feature Engineering

# todo: Predictive Model

# todo: Evaluation