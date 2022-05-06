import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from logisticRegression_R import read_dataset, preprocess_data, get_default_model
import matplotlib.pyplot as plt
import pickle
import argparse

parser = argparse.ArgumentParser(description="Parameters")
parser.add_argument('--vocab_sizes', type=str, default='500,1000,1500,2000,2500',
                    help='Vocabulary sizes (default: 500,1000,1500,2000,2500)')
args = parser.parse_args()
vocab_sizes = args.vocab_sizes
vocab_size_list = vocab_sizes.split(',')

dataset = read_dataset("./fake_or_real_news.csv")

#vocab_size_list = [500, 1000, 1500, 2000, 2500, 3000]
#vocab_size_list = [500, 1000, 1500, 2000, 2500, 3000]
acc_list = []
for vocab_size in vocab_size_list:
    X_train, X_test, y_train, y_test = preprocess_data(dataset, vocab_size = int(vocab_size))
    print(X_train.shape)
    lrClassifier = get_default_model()
    lrClassifier.fit(X_train, y_train)
    acc_list.append(accuracy_score(y_test, lrClassifier.predict(X_test)))

print(acc_list)
results = dict(vocab_size_list = vocab_size_list, acc_list = acc_list)
suffix = '_' + vocab_size_list[0] + '_' +vocab_size_list[-1]
with open('outputs/exp_vocab_size'+suffix+'.pkl', 'wb') as fout:
    pickle.dump(results, fout)

fig = plt.figure()
ax = fig.add_subplot()
ax.set_xlabel('Vocabulary sizes')
ax.set_ylabel('Accuracy')
plt.title('Vocabulary size experiment')
plotname = 'plots/exp_vocab_size' + suffix
x = [int(i) for i in vocab_size_list]
plt.plot(x, acc_list, color='red')
plt.savefig(plotname)
