# ds_CarPrice_Audi

# Data Science Audi Car Price Estimator : Project Overview

* Created a tool that estimates Used Car Prices of Audi's (in the UK), with a MAE of ~$4.1K, to give me a rough indication on whether or not I got a good deal for my Audi A3!!!.
* Took a selection of ~10K used Audis from https://www.kaggle.com/adityadesai13/used-car-dataset-ford-and-mercedes
* Engineered features from the text of each car description to quantify the value put on a vehicle being Diesel or Petrol, Automatic vs Manual transmission.
* Optimised Linear, Lasso and Random Forest Regressors using GridSearchCV to reach best model.


## Code and Resources Used


* Python version 3.7.6
* Packages: pandas, numpy, sklearn, matplotlib, seaborn, selenium, flask, json, pickle
* For web requirements: pip install -r requirements.txt
* For scraper, see below:
  * @author: Aditya
  * Kaggle : https://www.kaggle.com/adityadesai13/used-car-dataset-ford-and-mercedes
* Mark Down cheat sheet : https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet

## YouTube Walkthrough

* Used a tutorial from Ken Jee as a basis: online youtube tutorial
  * url : https://www.youtube.com/watch?v=MpF9HENQjDo&list=PL2zq7klxX5ASFejJj80ob9ZAnBHdz5O1t

## Data Cleaning
 
The data obtained was fairly clean, but i had to run a few commands to further categorise the data. The following changes were made and variables created:
  * Made columns for if different skills were listed in the job descriptions:
   * Diesel
   * Petrol
   * Hybrid
   * Manual
   * Semi-Automatic
   * Automatic
   * Split the Audis into models (A, T, S, R and Q series)
  
  
  ## EDA - Exploratory Data Analysis
  
  Looked at the distributions of the data and the value counts for the various categoriacal variables. Below are the highlights from the pivot tables.
  
  
  ## Model Building
  
  First, the categorical variables were transformed into dummy variables. The data was split into train and test sets with a test size of 30%.
  
  Three different models were tried, and were evaluated using Mean Absolute Error (MAE). MAE was chosen as it's relatively easy to interpret and outliers aren't particularly bad for this type of model.
  
  The three models tried were:
   * Multiple Linear Regression  - Baseline for the model
   * Lasso Regression - Because of the sparse data from the many categorical variables, a normalised regression like lasso was thought to be effective.
   * Random Forest - Again, with the sparsity associated with the data, this was thought to be a good fit.
  
  ## Model Performance
  
  The Random Forest model outperformed the other approaches on the test and validation sets.
  
   * Random Forest : MAE = 4.1
   * Linear Regression : MAE = 4.6
   * Ridge Regression : MAE = 4.6
  
  
  ## Additional Tests:
  
  Created a Jupyter Notebook (Audi_EDA_modeling.ipynb) where additional paramters were fed into the fitting. These were:
   * tax
   * mpg
   * engine size
   
   Using these additional paramters, the model performance was measured again for Multiple Linear Regression and Random Forrest.
   The data was split into train and test sets with a test size of 30%.
   
   * Linear Regression : MAE = 3.1 (R^2 = 0.84)
   * Random Forest : MAE = 1.7 (R^2 = 0.95)
   
   We see with the additional parameters, a significant improvement over the previous methodology run.
   
   ## Closing remarks
   
   I tested my own Audi within this machinary and I unfortunately got an unrealistic price (£3.6K, when the car would now be worth less than £1K, if that). I think to improve this, I would need more data at the lower price end (and higher age range). But this is a work in progress.
   
