import pandas as pd
import preprocess_utils as pu
from sklearn import preprocessing as sklearnpp
from sklearn.model_selection import train_test_split

def read_dataset(pathstring="./fake_or_real_news.csv"):
    dataset = pd.read_csv(pathstring)
    dataset['title_text'] = dataset['title'] + ' ' + dataset['text']
    return dataset

def preprocess_data(dataset, vocab_size = 1500):
    #cropped
    preprocessed_ds = pu.pre_processing(dataset['title_text'])
    X_data = pu.fit_vectorizer(preprocessed_ds, vec_type='tfidf', max_features = vocab_size)
    le = sklearnpp.LabelEncoder()
    le.fit(dataset['label'])
    y_binary = le.transform(dataset['label'])
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_binary, test_size = 0.2, random_state = 0)
    return X_train, X_test, y_train, y_test

def get_default_model():
    return LogisticRegression(max_iter = 10000)

def grid_search_lr(classifier, parameters = {
    'penalty': ['l2'],
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['newton-cg','lbfgs','liblinear','sag','saga']
    }):
    # penalty = ['l2']
    # C = [0.01, 0.1, 1, 10, 100]
    # solver = ['newton-cg','lbfgs','liblinear','sag','saga']
    # grid = dict(penalty=penalty,C=C,solver=solver)
    # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, n_jobs=-1, scoring='accuracy', verbose=2)
    return grid_search
