# Credit_Risk_Analysis

## Overview

### Evaluating Credit Worthiness
Determining credit worthiness has inherent problems as good loans easily outnumber risky loans.  It will, therefore, be necessary to deploy various techniques to train and evaluate models with unbalanced classes. 

### Project Task
imbalanced-learn and scikit-learn libraries will be used to build and evaluate models using resampling. Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, the following models will be applied:
* RandomOverSampler
* SMOTE algorithms
* ClusterCentroids algorithm
* SMOTEENN algorithm
* BalancedRandomForestClassifier
* EasyEnsembleClassifier

## Results

### Challenge to Any Model
![counts](https://github.com/cortesh/Credit_Risk_Analysis/blob/main/images/counts.jpg)

* The key fact of the data is the immense imbalance of the dependent variable, where high risk loans make up 0.5% of the total 68,470 valid records (the dataset contained a total of 115,674 records).  With such a lopsided data source, all models will be limited in their ability to successfully predict future high risk loans.

### Matrix of Competing Models
![matrix](https://github.com/cortesh/Credit_Risk_Analysis/blob/main/images/matrix_results.jpg)


## Summary

* Easy Ensemble AdaBoost Classifier out performed all other models beating out all models in every metric.  Particularly strong was its recall percentage or otherwise known as the test sensitivity.  While the precision for high risk predictions was low at 9%, it's likelihood of detecting high risk applicants was 92% (20 points higher than the average of all other models).  This disparity between precision and sensitivity raises some suspicions regarding the bias of this model.  Yet, it is far and away the highest performing, with an accuracy rate of 93%.

* The Undersampling model using the Cluster Centroids algorithm was the least performing model, contrary to initial expectations.  Undersampling methods are used to level off unbalanced majority portions to bring them within proportions of the smaller sample size.  As opposed to Oversampling techniques, this method does not generate artificial data to 'inflate' the minority sample size, and in theory should be more accurate.

* I would recommend, in addition to using the Easy Ensemble AdaBoost Classifier, that other analyses are used to limit outliers (such as box/whisker charts) as well as further studying the measures used and only importing those with the most demonstrated impact on the currrent selection of models.

