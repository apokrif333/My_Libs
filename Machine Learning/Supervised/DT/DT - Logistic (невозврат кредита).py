from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from graphviz import render

import os
import pandas as pd
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'


# Конвертируем dot в png
def dot_to_png(name: str):
    path = name + '.dot'
    render('dot', 'png', path)


# DecisionTreeClassifier
# Применяем дерево решений на синтетических данных. Создадим, таблицу с возрастом и невозвратом кредита.

data = pd.DataFrame({'Возраст': [17, 18, 20, 25, 29, 31, 33, 38, 49, 55, 64],
                     'Зарплата': [25, 22, 36, 70, 33, 102, 88, 37, 59, 74, 80],
                     'Невозрат кредита': [1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0]}
                    ).sort_values(by='Зарплата')

age_tree = DecisionTreeClassifier(random_state=17)
age_tree.fit(data['Возраст'].values.reshape(-1, 1), data['Невозрат кредита'].values)

export_graphviz(age_tree, feature_names=['Возраст'], out_file='age_tree.dot', filled=True)
dot_to_png('age_tree')

age_sal_tree = DecisionTreeClassifier(random_state=17)
age_sal_tree.fit(data[['Возраст', 'Зарплата']], data['Невозрат кредита'].values)

export_graphviz(age_sal_tree, feature_names=['Возраст', 'Зарплата'], out_file='age_sal_tree.dot', filled=True)
dot_to_png('age_sal_tree')

