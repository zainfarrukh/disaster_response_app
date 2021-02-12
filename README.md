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

#### 4. Files

Files are arranged in following way:<br/>
--App<br/>
----run.py <--RUNS THE MAIN APP<br/>
----templates <br/>
------go.html <--CLASSIFICATION RESULT PAGE OF WEB APP<br/>
------master.html <--MAIN PAGE OF WEB APP<br/>
--Data<br/>
----DisasterResponse.db <--DATABASE TO SAVED CLEANED DATA<br/>
----disaster_categories.csv <--DATA TO PROCESS<br/>
----disaster_messages.csv <--DATA TO PROCESS<br/>
----process.py <--SCRIPT TO PERFORM ETL PROCESS<br/><br/>
--Model<br/>
----train_classifier <--PERFORMS CLASSIFICATION TASKS<br/>

#### 5. How to run the App

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`<br/>
        The first two arguments are input data and the third argument is the SQLite Database in which we want to save the cleaned data. The ETL pipeline is in      process_data.py.
        DisasterResponse.db already exists in data folder but the above command will still run and replace the file with same information.<br/>

    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`<br/>
        This will use cleaned data to train the model, improve the model with grid search and saved the model to a pickle file (classifer.pkl).
        classifier.pkl already exists but the above command will still run and replace the file will same information.

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/ or link of wherever the file app is deployed

#### 6. Software Requirements
This project uses Python 3.6.3 and the necessary libraries are mentioned in requirements.txt. The standard libraries which are not mentioned in requirements.txt are json, operator, pickle, pprint, re, and sys.

#### 7. Acknowlegements
Credit goes to Figure Eight for giving me the data for this wonderful project and special thanks to Udacity for this awesome experience :)

