import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(train_path, test_path):
    train_data = pd.read_csv(train_path, delimiter = ';')
    test_data = pd.read_csv(test_path, delimiter = ';')
    return train_data, test_data

def preprocess_text(train_data, test_data):
    train_data['full_text'] = train_data['Summary'] + ' ' + train_data['Text']
    test_data['full_text'] = test_data['Summary'] + ' ' + test_data['Text']
    return train_data, test_data

def vectorize_data(train_data, test_data):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_data['full_text'])
    X_test = vectorizer.transform(test_data['full_text'])
    return X_train, X_test, vectorizer
