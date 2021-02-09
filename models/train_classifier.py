# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet'])

import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV

import pickle


def load_data(database_filepath):
    """
    Function to load the data from database and create X and Y features
    Inputs
    ------
    database_filepath: str
    File path of the db
    
    Returns
    -------
    X, Y features to be fed into the ML classifier
    
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('input_data', engine)
    X = df['message']
    y = df.drop(['id','message','original', 'genre'], axis=1).values
    label= df.drop(['id','message','original', 'genre'], axis=1).columns
    return X, y, label



def tokenize(text):
    """
    Function to text data and convert it into tokens
    Inputs
    ------
    text:str
    Test to be tokenized
    
    Returns
    -------
    Cleaned tokens
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    url_detector = re.findall(url_regex, text)
    for url in url_detector:
        text = text.replace(url, "urlplaceholder")
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens=[]
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    model = Pipeline([('count', CountVectorizer(tokenizer=tokenize)), ('tfid', TfidfTransformer()), 
                         ('cls', MultiOutputClassifier(RandomForestClassifier(min_samples_split=2)))])
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred_ = model.predict(X_test)
    Y_pred_ = pd.DataFrame(Y_pred_, columns=category_names)
    for i, var in enumerate(category_names):
        print(var)
        print (classification_report(Y_test[:,i], Y_pred_.iloc[:,i]))

def save_model(model, model_filepath):
    pickle.dump(model,open(model_filepath, 'wb'))
    

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()