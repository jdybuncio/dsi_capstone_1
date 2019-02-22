from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
import numpy as np



from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegressionCV
X, y = load_iris(return_X_y=True)
clf = LogisticRegressionCV(cv=5, random_state=0, multi_class='multinomial').fit(X, y)
clf.predict(X[:2, :])
#array([0, 0])
clf.predict_proba(X[:2, :]).shape
#(2, 3)
clf.score(X, y)
#0.98...


"""
ROSIE CODE

"""
def k_fold_CV(X, y, desc, n_folds = 5, cw = 'balanced', threshold = 0.5):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=False)
    auc_test = []
    accuracy_test = []
    precision_test = []
    recall_test = []
    f1_score_test = []
    auc_train = []
    accuracy_train = []
    precision_train = []
    recall_train = []
    f1_score_train = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.values[train_index], X.values[test_index]
        y_train, y_test = y.values[train_index], y.values[test_index]
        model = LogisticRegression(class_weight=cw)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        probabilities = np.where(probs >= threshold, 1, 0)
        probs_train = model.predict_proba(X_train)[:, 1]
        probabilities_train = np.where(probs_train >= threshold, 1, 0)
        auc_test.append(metrics.roc_auc_score(y_test, probs))
        accuracy_test.append(metrics.accuracy_score(y_test, probabilities))
        precision_test.append(metrics.precision_score(y_test, probabilities))
        recall_test.append(metrics.recall_score(y_test, probabilities))
        f1_score_test.append(metrics.f1_score(y_test, probabilities))
        auc_train.append(metrics.roc_auc_score(y_train, probs_train))
        accuracy_train.append(metrics.accuracy_score(y_train, probabilities_train))
        precision_train.append(metrics.precision_score(y_train, probabilities_train))
        recall_train.append(metrics.recall_score(y_train, probabilities_train))
        f1_score_train.append(metrics.f1_score(y_train, probabilities_train))
    return [np.mean(auc_test), np.mean(accuracy_test), np.mean(precision_test),np.mean(recall_test), np.mean(f1_score_test), np.mean(auc_train), np.mean(accuracy_train), np.mean(precision_train), np.mean(recall_train), np.mean(f1_score_train), desc]






















def roc_curve(probabilities, labels):
    '''
    INPUT: numpy array, numpy array
    OUTPUT: list, list, list

    Take a numpy array of the predicted probabilities and a numpy array of the
    true labels.
    Return the True Positive Rates, False Positive Rates and Thresholds for the
    ROC curve.
    '''

    thresholds = np.sort(probabilities)

    tprs = []
    fprs = []

    num_positive_cases = sum(labels)
    num_negative_cases = len(labels) - num_positive_cases

    for threshold in thresholds:
        # With this threshold, give the prediction of each instance
        predicted_positive = probabilities >= threshold
        # Calculate the number of correctly predicted positive cases
        true_positives = np.sum(predicted_positive * labels)
        # Calculate the number of incorrectly predicted positive cases
        false_positives = np.sum(predicted_positive) - true_positives
        # Calculate the True Positive Rate
        tpr = true_positives / float(num_positive_cases)
        # Calculate the False Positive Rate
        fpr = false_positives / float(num_negative_cases)

        fprs.append(fpr)
        tprs.append(tpr)

    return tprs, fprs, thresholds.tolist()


def run_fake_data():
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                               n_clusters_per_class=2, n_samples=1000)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]

    tpr, fpr, thresholds = roc_curve(probabilities, y_test)

    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity, Recall)")
    plt.title("ROC plot of fake data")
    plt.show()

def run_loan_data():
    df = pd.read_csv('data/loanf.csv')
    y = (df['Interest.Rate'] <= 12).values
    X = df[['FICO.Score', 'Loan.Length', 'Loan.Amount']].values
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]

    tpr, fpr, thresholds = roc_curve(probabilities, y_test)

    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity, Recall)")
    plt.title("ROC plot of loan data")
    plt.show()


if __name__ == '__main__':

    run_fake_data()
    run_loan_data()



from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.datasets import load_diabetes
import sklearn
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import pandas as pd
import matplotlib.pyplot as plt
import utils as ut

scaler = ut.XyScaler()
#
# In this assignment we will be comparing regularization methods on a well known dataset that is built into sklearn.
#
# ## Part 1: Loading The Data
#
# We're going to use a classic regression dataset for this example.
#
# 1. Load the diabetes data from sklearn using the instructions in the [sklearn documentation](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html).

diabetes = load_diabetes()

# 2. Some of the work at the end of this assignment will be very computationally heavy, so we will subset our data to make this work more approachable.  Take the first 100 rows from the diabetes data and target to use as your raw data in this assignment.
#

X = diabetes.data[:100]
y = diabetes.target[:100]

# 3. Take an initial look at the data and investigate what the predictors mean.  You may have to do some detective work with google.
#

# 10 baseline variables:
# age, sex, BMI, average blood pressure, and six blood serum measurements


# 4. Do some basic EDA.  Check for missing values, and plot the univariate and joint distributions of the predictors and target.  Make any sensible changes to the data based on what you discover in your explorations.

# should i normalize?

# ## Part 2: Ridge Regression
#
# Ridge regularization is a form of *shrinkage*: the parameter estimates are shrunk towards zero compared to the estimates from an unregularized regression. The amount of regularization (i.e. the severity of the shrinkage) is set via the `alpha` parameter of `Ridge`, which needs to be tuned with cross-validation.
#
# There is a `RidgeCV` class in sklearn which can automate some of the work involved in tuning alpha, but today we will do this manually to get our hands dirty.
#
# 1. Split the full data into a training and testing set.  The training set will be used to fit and tune all of your models, and the testing set will be used only at the very end to compare your very final models.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)



def std_data(mat):
    df_test = pd.DataFrame(mat)
    for i in df_test:
        df_test[i] = (df_test[i]-np.mean(df_test[i]))/np.std(df_test[i])

    return df_test.values


# #standardizing data

# 2. Let's fit a model just to get the mechanics of using `Ridge` down.  Fit a ridge regression with `alpha = 0.5` to your training dataset.  Use the fit model to generate predictions on your testing dataset.  Calculate the MSE of your fit model on the test set.

model_r = Ridge()
model_r.fit(X_train, y_train)
y_fit = model_r.predict(X_train)
np.sqrt(sklearn.metrics.mean_squared_error(y_fit, y_train))

#
# 3. Estimate the out of sample error of your ridge regression using 10-fold cross validation.  Remember that your predictors and response **must** be standardized when using ridge regression, and that this standardization must happen **inside** of the cross validation using **only** the training set!  Your code should look something like this:



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


def CV_kfold_ttsplit(X, y, base_estimator, n_folds, alpha, random_seed=154):

    kf = KFold(n_splits=n_folds, random_state=random_seed, shuffle=False)
    error_lst_test = []
    error_lst_train = []
    #split into test and train
    for train_index, test_index in kf.split(X):
        X_train_k, X_test_k = X[train_index], X[test_index]
        y_train_k, y_test_k = y[train_index], y[test_index]

        # Standardize data,
        X_train_k, y_train_k = scaler.fit(X_train_k, y_train_k).transform(X_train_k, y_train_k)
        X_test_k, y_test_k = scaler.fit(X_test_k, y_test_k).transform(X_test_k, y_test_k)
        #
        # X_train_k = std_data(X_train_k)
        # X_test_k = std_data(X_test_k)
        # y_train_k = std_data(y_train_k)
        # y_test_k = std_data(y_test_k)

        #fit on training set, transform training and test.
        model = base_estimator(alpha)
        model.fit(X_train_k, y_train_k)

        train_predicted = model.predict(X_train_k)
        test_predicted = model.predict(X_test_k)
        error_lst_test.append(sklearn.metrics.mean_squared_error(test_predicted, y_test_k))
        error_lst_train.append(sklearn.metrics.mean_squared_error(train_predicted, y_train_k))

    cv_test_mean = np.mean(error_lst_test)

    print(f'MSE for test set K-{n_folds}fold', cv_test_mean)

    return [cv_test_mean, alpha, n_folds]

# ```
#
# To make the task of standardizing both the predictor and response more seamless, we have provided an `XyScaler` class in `utils.py`.
#
# **Nota bene**: why standardize the response variable `y`? Normally we would leave `y` as it is, but due to a limitation of sklearn, this is the only way to ensure that the intercept term does not get regularized. For more discussion on why to leave the intercept term unregularized, see [here](https://stats.stackexchange.com/a/161689)
# alpha_list = list(np.arange(0.001, 30, 0.1))


cols = ['CV_mean_MSE', 'lambda', 'k-fold']
df = pd.DataFrame(columns = cols)

alpha_lass = [0.00001, 0.00002, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1, 1.5, 2, 5, 10, 15, 20, 50, 100]
for a in alpha_lass:
    df.loc[len(df)] = CV_kfold_ttsplit(X_train, y_train, Lasso, 10, a)


plt.plot(df['lambda'], df['CV_mean_MSE'])
plt.ylabel("Mean MSEs from CV (Lasso)")
plt.xlabel("Lambda Values")
# plt.vlines(21, 0.66, 0.71)
plt.xscale('log')
plt.show()
