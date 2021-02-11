# Disaster Response Pipeline Project
## Udacity Data Science Nanodegree Project

### Introduction:
This project takes in messages and categories csv files and trains the ML model to classify whether which category the message corresponds to. We will be creating a machine learning pipeline to categorize these messages so that we can send the messages to an appropriate disaster relief agency. 
Data has been provided by Figure Eight (https://appen.com/) and is copyright protected.

### Project Components
The project consists of three components

#### 1. ETL Pipeline
In a Python script, process_data.py, we wrote a data cleaning pipeline that:

1) Loads the messages and categories datasets
2) Merges the two datasets
3) Cleans the data
4) Stores it in a SQLite database

#### 2. ML Pipeline
In a Python script, train_classifier.py, wrote a machine learning pipeline that:

1) Loads data from the SQLite database
2) Splits the dataset into training and test sets
3) Builds a text processing and machine learning pipeline
4) Trains and tunes a model using GridSearchCV
5) Outputs results on the test set
6) Exports the final model as a pickle file

#### 3. Flask Web App
The Web App includes some of the visualizations of the training data and includes a prompt which take input in the form of a message from the user and classify that message in one or multiples of the 36 categories.
