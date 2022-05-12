## Description of the files
1. __LogisticRegression.ipynb__ : Main Jupyter Notebook containing all the experiments with Logistic regression.
2. __LogisticRegressionModules__ : Contains all the utility and text pre-processing methods.
3. __LSTM.ipynb__ : Main Jupyter notebook containing text classification using a single layer LSTM neural network.
4. __data/__: Contains the csv file of the [Fake or Real news dataset](https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news?resource=download) from Kaggle.
5. __plots/__: Contains the saved plots from the experiments.
6. __compare_LSTMS.py__: Prints out the performance numbers of the LSTM classifier with and without preprocessing(stop-word removal and stemming).
7. __scratch/__: Contains older code now not needed.
8. __Interpret_Negative_Result.csv__:
9. __Interpret_Positive_Result.csv__:
10. __interpret_test_v2.csv__:

## Reproducing the experiments
1. Download the news article datafile from [Kaggle](https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news?resource=download) and put it in _data/_ folder.
2. Open __LogisticRegression.ipynb__ and run all cells (the results should be reproducible).
3. Open __LSTM.ipynb__. Due to GPU-related randomness, the training of the LSTM will not be reproducible. Hence the performance numbers will change.

## System
1. For __LogisticRegression.ipynb__, we used our personal computers.
2. For __LSTM.ipynb__, we used Rivanna HPC with 30 core NVIDIA GPU with 6GB RAM.

## Interpreted Results
1. Words that contributed to the Positive Class (Real News):
  ![TrueWords](https://github.com/Luna-Xinyue/FakeNewsDetector/blob/main/wordcloud-positive.jpg)
2. Words that contributed to the Negative Class (Fake News):
  ![FalseWords](https://github.com/Luna-Xinyue/FakeNewsDetector/blob/main/wordcloud-negative.jpg)
