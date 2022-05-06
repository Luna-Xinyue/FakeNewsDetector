import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from logisticRegression_R import read_dataset, preprocess_data, get_model
import matplotlib.pyplot as plt

dataset = read_dataset("./fake_or_real_news.csv")

vocab_size_list = [500, 1000, 1500, 2000, 2500, 3000]
acc_list = []
for vocab_size in vocab_size_list:
    X_train, X_test, y_train, y_test = preprocess_data(dataset, vocab_size = vocab_size)
    lrClassifier = get_default_model()
    lrClassifier.fit(X_train, y_train)
    acc_list.append(accuracy_score(y_test, lrClassifier.predict(X_test)))


plt.plot(vocab_size_list, acc_list, color='red')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('Vocabulary sizes')
ax.set_ylabel('Accuracy')
plt.title('Vocabulary size experiment')
plotname = 'plots/exp_vocab_size'
plt.savefig(plotname)
