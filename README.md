# ds_CarPrice_Audi

# Data Science Audi Car Price Estimator : Project Overview

* Created a tool that estimates Used Car Prices of Audi's (in the UK), with a MAE of ~$4.1K, to give me a rough indication on whether or not I got a good deal for my Audi A3!!!.
* Took a selection of ~10K used Audis from https://www.kaggle.com/adityadesai13/used-car-dataset-ford-and-mercedes
* Engineered features from the text of each car description to quantify the value put on a vehicle being Diesel or Petrol, Automatic vs Manual transmission .
* Optimised Linear, Lasso and Random Forest Regressors using GridSearchCV to reach best model.
* Built a client facing API using Flask.

## Code and Resources Used


* Python version 3.7.6
* Packages: pandas, numpy, sklearn, matplotlib, seaborn, selenium, flask, json, pickle
* For web requirements: pip install -r requirements.txt
* For scraper, see below:
  * @author: arapfaik
  * GitHub : https://www.kaggle.com/adityadesai13/used-car-dataset-ford-and-mercedes
* Flask Productionise : https://towardsdatascience.com/productionize-a-machine-learning-model-with-flask-and-heroku-8201260503d2
* Mark Down cheat sheet : https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet

## YouTube Walkthrough

* Used a tutorial from Ken Jee as a basis: online youtube tutorial
  * url : https://www.youtube.com/watch?v=MpF9HENQjDo&list=PL2zq7klxX5ASFejJj80ob9ZAnBHdz5O1t

## Data Cleaning
 
The data obtained was fairly clean, but i had to run a few commands to take out it required cleaning so it was usable for the model. The following changes were made and variables created:
  * Made columns for if different skills were listed in the job descriptions:
   * Diesel
   * Petrol
   * Manual
   * Automatic
  
  
  ## EDA - Exploratory Data Analysis
  
  Looked at the distributions of the data and the value counts for the various categoriacal variables. Below are the highlights from the pivot tables.
  
  
  ## Model Building
  
  First, the categorical variables were transformed into dummy variables. The data was split into train and test sets with a test size of 20%.
  
  Three different models were tried, and were evaluated using Mean Absolute Error (MAE). MAE was chosen as it's relatively easy to interpret and outliers aren't particularly bad for this type of model.
  
  The three models tried were:
   * Multiple Linear Regression  - Baseline for the model
   * Lasso Regression - Because of the sparse data from the many categorical variables, a normalised regression like lasso was thought to be effective.
   * Random Forest - Again, with the sparsity associated with the data, this was thought to be a good fit.
  
  ## Model Performance
  
  The Random Forest model far outperformed the other approaches on the test and validation sets.
  
   * Random Forest : MAE = 4.1
   * Linear Regression : MAE = 18.7
   * Ridge Regression : MAE = 18.9
  
  ## Productionisation
In this step, a flask API endpoint that was hosted on a local webserver was built. This was done following a TDS tutorial in the reference section above. The API endpoint takes in a request with a list of values from a job listing and returns an estimated salary.
