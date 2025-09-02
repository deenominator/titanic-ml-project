Titanic Survival Prediction

This project is a classic data science challenge to predict whether a passenger on the Titanic survived or not. It uses a machine learning model to analyze passenger data and make predictions.

Project Overview

The goal of this project is to build a predictive model that accurately predicts the survival of passengers on the Titanic. The project involves the following steps:

Data Exploration and Visualization: Understanding the dataset, identifying key features, and visualizing their relationships with survival.

Data Preprocessing and Feature Engineering: Cleaning the data by handling missing values and converting categorical features into a numerical format suitable for machine learning.

Model Training: Training a RandomForestClassifier on the preprocessed training data.

Prediction: Making survival predictions on the test dataset.

Submission: Generating a submission.csv file in the required format for the Kaggle competition.

Dataset

The project uses the Titanic dataset from Kaggle, which is split into two files:

train.csv: Contains the training data along with the survival information (the ground truth).

test.csv: Contains the test data for which we need to predict survival.

Requirements

The following Python libraries are required to run this notebook:

pandas

numpy

matplotlib

seaborn

scikit-learn

You can install these dependencies using pip/anaconda:

pip install pandas numpy matplotlib seaborn scikit-learn

Usage

Clone this repository to your local machine.

Make sure you have the train.csv and test.csv files in the same directory as the notebook.

Open and run the titanic prediction.ipynb notebook in a Jupyter environment.

The notebook will process the data, train the model, and generate a submission.csv file with the predictions.

Model Performance

The RandomForestClassifier model achieved a validation accuracy of 82.12%.

The features used for training the model were:

Pclass

Sex

Fare

Age

Embarked
