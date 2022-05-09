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
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def pre_processing(dataset):
    #Stop Words Removal and stemmer
    ps = PorterStemmer()
    stopwords = nltk.corpus.stopwords.words('english')
    tokens_after_sw_and_ps = []
    preprocessed_dataset = []
    tokens_after_sw = []
    text_tokens = []
    for i in range(dataset_title.size):
        tokens_after_sw_and_ps.clear()
        tokens_after_sw.clear()
        text_tokens.clear()
        text_tokens = word_tokenize(dataset[i])
        tokens_after_sw = [word for word in text_tokens if not word in stopwords]
        for w in tokens_after_sw:
            tokens_after_sw_and_ps.append(ps.stem(w))
        preprocessed_dataset.append((" ").join(tokens_after_sw_and_ps))
    return preprocessed_dataset

def fit_vectorizer(data_set, vec_type="binary", max_features = 1500):
    output = []
    if "binary" in vec_type:
        cv = CountVectorizer(binary=True, max_df=0.95, max_features = max_features)
        output = cv.fit_transform(data_set).toarray()
        
    if "counts" in vec_type:
        cv = CountVectorizer(binary=False, max_df=0.95, max_features = max_features)
        output = cv.fit_transform(data_set).toarray()

    elif "tfidf" in vec_type:
        tfidf = TfidfVectorizer(use_idf=True)
        output = tfidf.fit_transform(data_set).toarray()
    return output


# Read data
dataset = pd.read_csv('../input/fake-or-real-news/fake_or_real_news.csv')
dataset_title = dataset['title']

preprocessed_dataset = pre_processing(dataset_title)

X_data = fit_vectorizer(preprocessed_dataset, vec_type="tfidf", max_features = 1500)

# X = preprocessing.normalize(X_data, norm='l2', axis=1, copy=True, return_norm=False)

le = preprocessing.LabelEncoder()
le.fit(dataset['label'])
y_binary = le.transform(dataset['label'])

# Dataset split (0.7/0.3)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_binary, test_size = 0.2, random_state = 0)

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
