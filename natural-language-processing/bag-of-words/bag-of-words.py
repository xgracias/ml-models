import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Temporary fix: Disable SSL verification (not recommended for security-sensitive applications)
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
print(dataset)

# cleaning texts
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words(('english')))]
    review = ' '.join(review)
    corpus.append(review)

print(corpus)