import warnings; warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# Logistic Regression
def logistic_regression_accuracy(dataframe, labels):
    features = dataframe.as_matrix()
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels)

    logit = LogisticRegression()
    logit.fit(train_features, train_labels)
    return classification_report(test_labels, logit.predict(test_features))


df = pd.read_csv('D:/PycharmProjects/Start/GitHub/My_Libs/ml_examples/test_data/bank_train.csv')
labels = pd.read_csv('D:/PycharmProjects/Start/GitHub/My_Libs/ml_examples/test_data/bank_train_target.csv', header=None)

# df['education'].value_counts().plot.barh()
# plt.show()

# LabelEncoder
label_encoder = LabelEncoder()
mapped_education = pd.Series(label_encoder.fit_transform(df['education']))
# mapped_education.value_counts().plot.barh()
# plt.show()
print(dict(enumerate(label_encoder.classes_)))
df['education'] = mapped_education

categorical_columns = df.columns[df.dtypes == 'object'].union(['education'])
for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])

print(logistic_regression_accuracy(df[categorical_columns], labels))

# One-Hot Encoding. Хотя обычно, из-за большого размера создаётся раряженная матрица
onehot_encoder = OneHotEncoder(sparse=False)
encoded_categorical_columns = pd.DataFrame(onehot_encoder.fit_transform(df[categorical_columns]))
print(logistic_regression_accuracy(encoded_categorical_columns, labels))

# Hashing trick
print('---------------------------')
hash_space = 25
for s in ('university.degree', 'high.school', 'illiterate'):
    print(s, '->', hash(s) % hash_space)

print('---------------------------')
hashing_example = pd.DataFrame([{i: 0.0 for i in range(hash_space)}])
for s in ('job=student', 'marital=single', 'day_of_week=mon'):
    print(s, '->', hash(s) % hash_space)
    hashing_example.loc[0, hash(s) % hash_space] = 1
print(hashing_example)
