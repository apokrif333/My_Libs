from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
from graphviz import render

import os
import pandas as pd
import numpy as np
import ml_lib
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'

df = pd.read_csv('test_data/Breast Cancer Wisconsin_train.csv')

y_train  = df['Class'].replace(('benign', 'malignant'), (0, 1))
df = df.drop('Class', axis=1)
X_train = df.values

clf_tree_tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=17)
clf_tree = clf_tree_tree.fit(X_train, y_train)
# export_graphviz(clf_tree, feature_names=df.columns, out_file='img/Alex.dot', filled=True,  class_names=['benign', 'malignant'])
# ml_lib.dot_to_png('Alex')

print(cross_val_score(clf_tree_tree, X_train, y_train, cv=5))

df_test = pd.read_csv('test_data/Breast Cancer Wisconsin_test.csv')

y_test  = df_test ['Class'].replace(('benign', 'malignant'), (0, 1))
df_test = df_test .drop('Class', axis=1)
X_test = df_test .values

print(accuracy_score(y_test, clf_tree.predict(X_test)))

