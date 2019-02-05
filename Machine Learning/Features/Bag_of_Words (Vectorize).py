from functools import reduce

import numpy as np


def vectorize(text):
    vector = np.zeros(len(dictionary))
    for i, word in dictionary:
        num = 0
        for w in text:
            if w == word:
               num += 1

        if num:
            vector[i] = num

    return vector


texts = [
    ['i', 'have', 'a', 'cat'],
    ['he', 'have', 'a', 'dog'],
    ['he', 'and', 'i', 'have', 'a', 'cat', 'and', 'a', 'dog']
]

dictionary = list(enumerate(set(reduce(lambda x, y: x + y, texts))))
print(dictionary)

for t in texts:
    print(t)
    print(vectorize(t))
