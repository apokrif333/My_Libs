from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import euclidean


# ----------------------------------------------------------------------------------------------------------------------
vect = CountVectorizer(ngram_range=(1, 1))
vect_toarray = vect.fit_transform(['no i have cows', 'i have no cows']).toarray()
print(vect.vocabulary_)
print(vect_toarray)

vect = CountVectorizer(ngram_range=(1, 2))
vect_toarray = vect.fit_transform(['no i have cows', 'i have no cows'])
print(vect.vocabulary_)
print(vect_toarray)
# ----------------------------------------------------------------------------------------------------------------------

vect = CountVectorizer(ngram_range=(3, 3), analyzer='char_wb')
n1, n2, n3, n4 = vect.fit_transform(['иванов', 'петров', 'петренко', 'смит']).toarray()
print("----------")
print(vect.vocabulary_)
print(n1)

print(euclidean(n1, n2))
print(euclidean(n2, n3))
print(euclidean(n3, n4))
