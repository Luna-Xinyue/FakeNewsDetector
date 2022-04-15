import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse import data
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# Read data
dataset = pd.read_csv('../input/fake-or-real-news/fake_or_real_news.csv')
dataset_title = dataset['title']

# pre-processing
#Stop Words Removal and stemmer
ps = PorterStemmer()
stopwords = nltk.corpus.stopwords.words('english')
# print(dataset_title.head())

tokens_without_sw_stem = []
dataset_title_after = []
tokens_without_sw = []
text_tokens = []
for i in range(dataset_title.size):
    tokens_without_sw_stem.clear()
    tokens_without_sw.clear()
    text_tokens.clear()
    text_tokens = word_tokenize(dataset_title[i])
    tokens_without_sw = [word for word in text_tokens if not word in stopwords]
    for w in tokens_without_sw:
        tokens_without_sw_stem.append(ps.stem(w))
    dataset_title_after.append((" ").join(tokens_without_sw_stem))

# print(dataset_title.head())
# Binarize the dataset
cv = CountVectorizer(max_features = 1500)
X_title = cv.fit_transform(dataset_title_after).toarray()

X = preprocessing.normalize(X_title, norm='l2', axis=1, copy=True, return_norm=False)

le = preprocessing.LabelEncoder()
le.fit(dataset['label'])
y_binary = le.transform(dataset['label'])

# Dataset split (0.7/0.3)
X_train, X_test, y_train, y_test = train_test_split(X_title, y_binary, test_size = 0.3, random_state = 0)

# Train the logistic regression classifier
classifier = LogisticRegression(penalty='l2', fit_intercept=True, C=1)
classifier.fit(X_train, y_train)

# Using the classifier to predict and evaluate the classfier
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
TP = cm[1][1]
TN = cm[0][0]
FP = cm[1][0]
FN = cm[0][1]

Accuracy = (TP + TN) / (TP + TN + FP + FN) 
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1_Score = 2 * Precision * Recall / (Precision + Recall)

print(Accuracy, Precision, Recall, F1_Score)
