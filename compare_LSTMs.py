import pandas as pd

no_pp_results = pd.read_pickle('./outputs/LSTM_results_no_preprocessing.pkl')
pp_results = pd.read_pickle('./outputs/LSTM_results_with_preprocessing.pkl')
print('LSTM Results with no preprocessing')
print(f"Test accuracy:{no_pp_results['accuracy']}\nTest F1-score:{no_pp_results['f1_score']}")
print('--------------\nLSTM Results with preprocessing')
print(f"Test accuracy:{pp_results['accuracy']}\nTest F1-score:{pp_results['f1_score']}")
