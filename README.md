# bankrupty-classification-eda
The research question of interest is the predict if a firm in a given year has entered bankruptcy proceedings (1) or has not (0). 

The code can be run interactively through a [Google Colab Notebook]().

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

