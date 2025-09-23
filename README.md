# Credit_Card_Fraud_Detection
An investigation of the Credit Card Fraud Detection data set on Kaggle.

The original data set and a summary of it's input variables etc can be found here:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Goal:

Investigate the data set, potentially flag any problems, and eventually try to create a ML/AI model that predicts fraudulent financial transactions.

## Navigating the Repository

The code I have written for all my analysis is in:

fraud_detection_initial_screen.ipynb

which can be found in the main repository.

### The folder breakdown_plots

Contains a variety of plots I created.  Mostly histograms.

There are 2 main folders to look at.

Histograms: Contains the histograms for each variable provided in the data set.  This can technically already be found on the Kaggle data set, but I wanted to do the analysis myself.

Histograms_Downsample_Split: Here I have done 2 things. First I downsampled the majority class (non-fraudulent) so that there is the same number of fraudulent and non-fraudulent cases. These histograms also show the separate histograms for both classes:

0: Non-Fraudulent Transaction

1: Fraudulent Transaction

## Initial observations:

The data set only covers 2 days of data, and does not include separation of clients.  These combined mean we can't make any longer term pattern recognition of how particular clients tend to behave.

The data set has been altered from it's original format.  Most of the given variables are from Principle Component analysis (PCA).  I would hazard a guess that these PCA components are likely from original variables that provide categories such as but not excluding:
-transaction type
-country of money origin
-country of money destination
-business or personal accounts
-likely other variables I'm not aware of within the financial space

The way I would have done this if I had these variables is that I would have one-hot-encoded the variables just listed.  Before one-hot-encoding, I'd check for data quality etc and see if the data needed cleaning.  I would also feature engineer things like turning country of origin into things like international vs national transactions.

After one-hot-encoding, I would reduce these features using PCA (a feature extraction methodology using orthogonality).

The reason for doing this is that the one-hot-encoded data set was likely way too large in scale.  It would create 2 problems, first being the time it would take a model to train, and another being the model's ability to parse out the important information.  PCA has the ability to highlight important/differentiating features.

## Initial Analysis

This data set looks pretty clean at a first glance.  No missing values. 31 features (including Class) and 284807 total cases.

The data is highly imbalanced as one can see here:

Number of non-fraudulent transactions: 284315

Number of fraudulent transactions: 492

Fraudulent transactions make up a very small 0.17% of the data.
