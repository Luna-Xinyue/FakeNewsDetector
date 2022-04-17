# FakeNewsDetector

## Luna: Uploaded first version of logistic regression.
  - This demo shows that this is a plausible method. 
  - I have created a code structure can be used later
### Todo List
  - [ ] re-structure the code, add input function to using different text as training set
  - [ ] Text pre-process

## Rick: Uploaded first version of LSTM.
- I first prepared the dataset from the Kaggle Fake or real csv file.
- I built a sequential model using tensorflow/keras, with a single layer LSTM.
- The training time for 1 epoch on an 80-20 split, is about 22 min.
- Note that this is a first version.

## Rick: Uploaded second version of LSTM.
- This time I used a Rivanna cluster of 2 GPUs. Training took about 3 min/epoch.
- Ran for 10 epochs.
## Rick: To Do:
- [ ] Try Early Stopping
- [ ] Try with one-hot labels and 2 output units.
- [ ] Grid Search optimization

## Mohamed: Uploading Grid Search for Logistic Regression
  - Built on Luna's Logistic Regression code
  - Using the normalized input
  - Grid Search on values of C and solver, with penalty set to 'l2'
## Morteza: Add Pre-processing to the Logistic Regression Code
  - Add Stop Word Removal 
  - Add Porter Stemmer
## Morteza: Add two modules, TF-IDF, and "counts" vectorizer
  - Add two modules: pre-processing and fit_vectorizer
  - Add tf-idf and counts vectorizers

## Mohamed: 
  - Updated LogisticRegression_Grid.py
  - Combines Logistic Regression + Cropping + Pre-Processing +Grid Search