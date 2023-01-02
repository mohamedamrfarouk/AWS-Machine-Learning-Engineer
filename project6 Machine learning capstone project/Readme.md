
## Project Overview :
This data set contains simulated data that mimics customer behavior on the Starbucks rewards mobile app. Once every few days, Starbucks sends out an offer to users of the mobile app. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). Some users might not receive any offer during certain weeks.
Not all users receive the same offer, and that was the challenge to solve with this data set.
Our task is to combine transaction, demographic and offer data to determine which demographic groups respond best to which offer type. This data set is a simplified version of the real Starbucks app because the underlying simulator only has one product whereas Starbucks actually sells dozens of products.

## Problem Statement :
Predicting the purchase offer will complete or no

## Installations
This project was written in Python, using Jupyter Notebook on AWS sagemaker notebook instance. 
The needed Python packages and libraries for this project are:

- pandas
- numpy
- math
- json
- matplotlib
- sklearn
- aws sagemaker libraries 


## File Descriptions
contains :
-  `the project proposal`: reviewed and accepted before https://review.udacity.com/#!/reviews/3892116
-  `Starbucks_Capstone_notebook.ipynb` : the code notebook .
-  `Capstone-Project-Report.pdf` : my project report .
-  `createendpoint.ipynb` : the AWS deployment
-  `Data` :
    - 1. Profile.json
    - 2. Portfolio.json
    - 3. transcript.json
    - 4. train.csv
    - 5. test.csv
