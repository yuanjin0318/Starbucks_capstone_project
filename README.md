# Starbucks Capstone Challenge
Project in Data Scientist Nanodegree of Udacity


## Project Motivation<a name="motivation"></a>

It is the Starbuck's Capstone Challenge of the Data Scientist Nanodegree in Udacity. We get the dataset from the program that creates the data simulates how people make purchasing decisions and how those decisions are influenced by promotional offers. We want to make a recommendation engine that recommends Starbucks which offer should be sent to a particular customer.

In this data analysis project, I will try to solve the following questions
(1). Which demographic groups respond best to which offer type? What is the best offer strategy for Starbucks?
(2). How possible will a customer use the offer sent to him or her? Are there any common characteristics of the customers who take the offer?

In this data analysis, since we are going to investigate the impact of each offer type on the customers. My strategy for solving this problem has four steps. First, I will combine the offer portfolio, customer profile, and transaction data. Each row of this combined dataset will describe an offer's attributes, customer demographic data, and whether the offer was successful. Second, I will assess the accuracy to juedge my model. This provides me a baseline for evaluating the performance of models that I construct. Accuracy measures how well a model correctly predicts whether an offer is successful. In other words, we will use accuracy as the main metric to understand the pro and cons of our machine learning model, which is much more intuitive and direct for the decision-makers compared to other metrics like F-1 score.

## libraries used 

import pandas as pd
import numpy as np
import math
import json
import matplotlib.pyplot as plt
import seaborn as sb 
import warnings
import statsmodels.api as sm
from datetime import datetime
from time import time

from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,f1_score, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split,GridSearchCV, cross_validate
from sklearn import metrics


## File Descriptions <a name="files"></a>

The notebook available here showcases work related to the above questions.  

This data set is a simplified version of the real Starbucks app because the underlying simulator only has one product whereas Starbucks actually sells dozens of products.

The data is contained in three files:
- `portfolio.json` - containing offer ids and meta data about each offer (duration, type, etc.)
- `profile.json` - demographic data for each customer
- `transcript.json` - records for transactions, offers received, offers viewed, and offers completed

Here is the schema and explanation of each variable in the files:

`portfolio.json`
- id (string) - offer id
- offer_type (string) - the type of offer ie BOGO, discount, informational
- difficulty (int) - the minimum required to spend to complete an offer
- reward (int) - the reward is given for completing an offer
- duration (int) - time for the offer to be open, in days
- channels (list of strings)

`profile.json`
- age (int) - age of the customer
- became_member_on (int) - the date when customer created an app account
- gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
- id (str) - customer id
- income (float) - customer's income

`transcript.json`
- event (str) - record description (ie transaction, offer received, offer viewed, etc.)
- person (str) - customer id
- time (int) - time in hours since the start of the test. The data begins at time t=0
- value - (dict of strings) - either an offer id or transaction amount depending on the record


## Files
Starbucks_Capstone_notebook.ipynb: the code notebook
data : the original data file 
temporary.csv: temporaray file with the customers response to the Starbucks offers 
README.md readme file

## Results<a name="results"></a>

The main findings of the code can be found at the post available [here](https://medium.com/@yuanjin0318/starbucks-rewards-offer-analysis-b2895898a99c).

Based on the transcript records, I build a Decision Tree Classifier based model to analysis the customers response to each type of reward offer. 


