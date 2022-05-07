from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from nltk.stem import PorterStemmer
import nltk
from nltk.tokenize import word_tokenize

def pre_processing(dataset):
    #Stop Words Removal and stemmer
    ps = PorterStemmer()
    stopwords = nltk.corpus.stopwords.words('english')
    tokens_after_sw_and_ps = []
    preprocessed_dataset = []
    tokens_after_sw = []
    text_tokens = []
    for i in range(dataset.size):
        tokens_after_sw_and_ps.clear()
        tokens_after_sw.clear()
        text_tokens.clear()
        text_tokens = word_tokenize(dataset[i])
        tokens_after_sw = [word for word in text_tokens if not word in stopwords]
        for w in tokens_after_sw:
            tokens_after_sw_and_ps.append(ps.stem(w))
        preprocessed_dataset.append((" ").join(tokens_after_sw_and_ps))
    return preprocessed_dataset


def fit_vectorizer(dataset, vec_type="binary", max_features = 1500):
    output = []
    if "binary" in vec_type:
        cv = CountVectorizer(binary=True, max_df=0.95, max_features = max_features)
        output = cv.fit_transform(dataset).toarray()
    elif "counts" in vec_type:
        cv = CountVectorizer(binary=False, max_df=0.95, max_features = max_features)
        output = cv.fit_transform(dataset).toarray()
    elif "tfidf" in vec_type:
        tfidf = TfidfVectorizer(use_idf=True, max_df=0.95, max_features = max_features)
        output = tfidf.fit_transform(dataset).toarray()

    output = preprocessing.normalize(output, norm='l2', axis=1, copy=True, return_norm=False)
    return output

def fit_vectorizer_v2(dataset, vec_type="binary", max_features = 1500):
    output = []
    if "binary" in vec_type:
        cv = CountVectorizer(binary=True, max_df=0.95, max_features = max_features)
        output = cv.fit_transform(dataset).toarray()
        vocab = None
    elif "counts" in vec_type:
        cv = CountVectorizer(binary=False, max_df=0.95, max_features = max_features)
        output = cv.fit_transform(dataset).toarray()
        vocab = None
    elif "tfidf" in vec_type:
        tfidf = TfidfVectorizer(use_idf=True, max_df=0.95, max_features = max_features)
        output = tfidf.fit_transform(dataset).toarray()
        vocab = tfidf.vocabulary_

    output = preprocessing.normalize(output, norm='l2', axis=1, copy=True, return_norm=False)
    return output, vocab

def crop(raw_text, perc):
    escapes = ''.join([chr(char) for char in range(1, 32)])
    translator = str.maketrans('', '', escapes)
    clean_text = raw_text.translate(translator)
    split_text = clean_text.split(' ')
    total_text_length = len(split_text)
    new_text_length = int(perc*total_text_length)
    split_text = split_text[0:new_text_length]
    cropped_text = ' '.join(split_text)
    return cropped_text

def run_default_logistic_regression(X_data, y_binary):
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_binary, test_size = 0.2, random_state = 0)
    
    clf = LogisticRegression(max_iter=10000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    TP = cm[1][1]
    TN = cm[0][0]
    FP = cm[1][0]
    FN = cm[0][1]

    Accuracy = (TP + TN) / (TP + TN + FP + FN) 
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1_Score = 2 * Precision * Recall / (Precision + Recall)

    print('Accuracy:' + str(Accuracy) + ', Precision:' + str(Precision) 
          + ', Recall:' + str(Recall)+ ', F1_Score:' + str(F1_Score))
    return Accuracy