import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

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

def crop(raw_text):
    raw_text_split = raw_text.split('.')
    escapes = ''.join([chr(char) for char in range(1, 32)])
    translator = str.maketrans('', '', escapes)
    raw_text_split_no_escape = []
    for part in raw_text_split[0:20]:
        part_clean = part.translate(translator)
        raw_text_split_no_escape.append(part_clean)
    cropped_text = ' '.join(raw_text_split_no_escape)
    return cropped_text
