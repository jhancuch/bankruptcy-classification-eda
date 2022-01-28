# bankrupty-classification-eda
The research question of interest is the predict if a firm in a given year has entered bankruptcy proceedings (1) or has not (0). 

The code can be run interactively through a [Google Colab Notebook](https://colab.research.google.com/github/jhancuch/bankruptcy-classification-eda/blob/main/bankrupty-classification-eda.ipynb).

## Research design and modeling methods
The research design uses three different classifiers, Support Vector Machine (SVM), Logistic Regression (Logit) and Naive Bayes. Examining the data through EDA, I discover I have an unbalanced dataset with only ~3% of observations being bankrupt firms. Thus, when I conduct hyperparameter tuning for SVM and Logit through a grid search with a cross validation of k=5, I will include different class weights as parameters to help determine the optimal model parameters. For Naive Bayes, instead of the standard model, I use the Complement Naive Bayes model to help tackle the unbalanced dataset. 

I additionally utilize a standard scaler for SVM and Logit but use a min-max scaler for Naive Bayes since the inputs must be positive. 

## Results and evaluation
| Metric | SVM | Logit | Complement Naive Bayes |
|---     | --- | ---   |---                     |
| Validation Accuracy | 0.97 | 0.97 | 0.52 |
| Validation TPR | 0.05 | 0.11 | 0.81 |
| Validation TNR | 1.00 | 0.99 | 0.51 |
| Validation (1) Precision | 0.25 | 0.33 | 0.04 |
| Validation (1) Recall | 0.05 | 0.11 | 0.81 |
| Validation (1) F1-Score | 0.09 | 0.16 | 0.08 |
| Validation (0) Precision | 0.97 | 0.98 | 0.99 |
| Validation (0) Recall | 1.00 | 0.99 | 0.51 |
| Validation (0) F1-Score | 0.98 | 0.98 | 0.68 |

Overall, the model development was underwhelming. Between SVM, Logit, and the Complement Naive Bayes, Logit is the best model. Complement Naive Bayes takes itself out of the running with its ppor precision and recall of bankrupt firms. Logit narrowly beats out SVM due to its marginal increase in precision and recall for bankrupt firms and a only 1% decline in recall of non-bankrupt firms compared to SVM. We also see that Logit has the highest F1-Score for bankrupt firms and non-bankrupt firms.

## Discusion
A potential area of improvement lies in variable selection. No variable selection was utilized but potentially using correlation heat maps and a random forest based algorithm to determine the importance of each variable could prove useful. Potentially the models have too much noise with extraneous variables leading to less than optimal outcomes.
