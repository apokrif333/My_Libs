import warnings; warnings.filterwarnings('ignore')
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_20newsgroups, load_files
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve, confusion_matrix
from scipy.sparse import csr_matrix


# Приведём данные в VW-формат
def to_vw_format(document, label=None):
    return str(label or '') + ' |text ' + ' '.join(re.findall('\w{3,}', document.lower())) + '\n'


newsgroups = fetch_20newsgroups('D:/PycharmProjects/Start/GitHub/My_Libs/ml_examples/test_data/news_data')
text = newsgroups['data'][0]
target = newsgroups['target_names'][newsgroups['target'][0]]
print('-------', '\n',
      text.strip())
print('-------', '\n',
      target)
print(to_vw_format(text, 1 if target == 'rec.autos' else -1))

# Присвоим целевой вектор - новости про авто
all_documents = newsgroups['data']
all_targets = [1 if newsgroups['target_names'][target] == 'rec.autos' else -1 for target in newsgroups['target']]
train_doc, test_doc, train_labels, test_labels = train_test_split(all_documents, all_targets, random_state=7)

with open('D:/PycharmProjects/Start/GitHub/My_Libs/ml_examples/test_data/news_data/20news_train.vw', 'w') as \
        vw_train_data:
    for text, target in zip(train_doc, train_labels):
        vw_train_data.write(to_vw_format(text, target))

with open('D:/PycharmProjects/Start/GitHub/My_Libs/ml_examples/test_data/news_data/20news_test.vw', 'w') as \
        vw_train_data:
    for text in test_doc:
        vw_train_data.write(to_vw_format(text))

# https://pypi.org/project/vowpalwabbit/