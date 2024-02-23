import warnings
warnings.filterwarnings('ignore')
from IPython.display import display
from matplotlib import pyplot as plt
from matplotlib import style
style.use(style='seaborn-v0_8-deep')
from tabulate import tabulate
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns
import os
os.chdir('D:\Stellar-Classification-Problem-main(1)\Stellar-Classification-Problem-main')
data_df = pd.read_csv(filepath_or_buffer='star_classification.csv')

print(data_df.head())
# Metadata of the dataset.

data_df.info()

print("The shape of the dataset: {}".format(data_df.shape))

# Class distribution of the dataset.

labels = data_df['class'].unique()
print(labels)
# Select only the specified columns
columns_to_keep = ['delta', 'alpha', 'u', 'g', 'r', 'i', 'z', 'class', 'redshift']
data_df = data_df[columns_to_keep]

# List of columns for which to plot distributions, excluding 'class'
columns_to_plot = [col for col in columns_to_keep if col not in ['class']]

# Determine the size of the grid
n = len(columns_to_plot)
ncols = 3  # Number of columns in the grid
nrows = n // ncols + (n % ncols > 0)  # Calculate rows needed, add one if there's a remainder

# Create a figure and axes for the subplots
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows * 5))
fig.tight_layout(pad=5.0)

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Plot distributions of numerical columns with respect to 'class'
for i, col in enumerate(columns_to_plot):
    sns.boxplot(ax=axes[i], x='class', y=col, data=data_df)
    axes[i].set_title(f'Distribution of {col} across Classes')
    axes[i].set_ylabel(col)
    axes[i].set_xlabel('Class')

# Hide any unused axes if the number of plots is less than the number of subplots
for j in range(i + 1, nrows * ncols):
    axes[j].set_visible(False)

plt.show()

# From the above distribution and proportion plots, I see that the dataset is not balanced across the target variable.

# Checking for `NULL` values in the dataset.

display(data_df.isnull().sum())

# This dataset has zero NULL values.

# __3.3. Selecting important features from domain understanding__

# From section 2.2, I understand that some features in the dataset are significantly useful such as: navigation angles - _ascension_ and _declination_, filters of the photometric system - _u_, _g_, _r_, _i_, _z_, and _redshift_. All other columns in the dataset are IDs.

imp_cols = ['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'redshift']

# Basic statistics on selected columns.

display(data_df[imp_cols].describe().T)

# __3.4. Boxplots for univariate analysis__

def box_plotter(df, features, target):
    """
    This is funtion helps in plotting the boxplot of data.
    
    Parameters
    ----------
    `df`: dataset
    `features`: columns for analysis
    `target`: target column name
    
    Returns none.
    """
    plt.figure(figsize=(15, 30))
    for (i, feature) in zip(range(len(features)), features):
        plt.subplot(int('{}2{}'.format(len(features), i+1)))
        sns.boxplot(x=target, y=feature, data=df,
                    width=0.5, flierprops={'marker': 'x'})
        plt.title(label='Boxplot of {}'.format(feature), fontsize=10)
        plt.xlabel(xlabel=None)
        plt.ylabel(ylabel=None)
        i += 1
    plt.show()

box_plotter(df=data_df, features=imp_cols, target='class')

# From the above boxplots, it is clear that columns - _u_, _g_, and _z_ have one outlier value that belongs to the __STAR__ class. The outlier value is located in the $79543$ index of the dataset.

display(data_df[data_df['u'] == min(data_df['u'])][imp_cols + ['class']])

display(data_df[data_df['g'] == min(data_df['g'])][imp_cols + ['class']])

display(data_df[data_df['z'] == min(data_df['z'])][imp_cols + ['class']])

# Removing the outlier index from the dataset.

data_df = data_df.drop(index=[79543])
print("The shape of the dataset: {}".format(data_df.shape))

box_plotter(df=data_df, features=imp_cols, target='class')

# The boxplot of the columns - _u_, _g_, and _z_, looks better as I removed the outlier value from the dataset. However, when I observe the box plot of the _redshift_ column, I see that the redshift values for the __STAR__ class are almost $0$.

# __3.5. PDF plots for univariate analysis__

def pdf_plotter(df, features, target):
    """
    This is funtion helps in plotting the pdf of data.
    
    Parameters
    ----------
    `df`: dataset
    `features`: columns for analysis
    `target`: target column name
    
    Returns none.
    """
    plt.figure(figsize=(15, 30))
    for (i, feature) in zip(range(len(features)), features):
        plt.subplot(int('{}2{}'.format(len(features), i+1)))
        sns.kdeplot(data=df, x=feature, hue=target, shade=True)
        plt.title(label='PDF of {}'.format(feature), fontsize=10)
        plt.xlabel(xlabel='')
        plt.grid()
        i += 1
    plt.show()

pdf_plotter(df=data_df, features=imp_cols, target='class')

# From the above density plots, I see that the density of all columns is overlapping, except for the _redshift_ column. In the density plot of the _redshift_ column, the __STAR__ class has all $0$ values. The kernel density plots of the columns do not follow the Gaussian distribution.



# __4.1. Data splitting__

# I will split the dataset into 3 sets - Train, Cross validation, and Test sets. This is a good practice to test the model's performance before deploying it in the production.
# 
# - Train set will have $60\%$ of the data.
# - Cross validation set will have $20\%$ of the data.
# - Test set will have $20\%$ of the data.

imp_cols = ['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'redshift']

X = data_df[imp_cols]
y = data_df['class'].values

print(X.columns)
print("The shape of X: {}".format(X.shape))
print("The shape of y: {}".format(y.shape))

from sklearn.model_selection import train_test_split

# As the dataset I have currently is imbalanced, I need to split the dataset by applying _stratified sampling_ which keeps the diversity of target variable intact.

(X_train,
 X_test,
 y_train,
 y_test) = train_test_split(X, y,
                            stratify=y, test_size=0.20, random_state=0)

(X_train,
 X_cv,
 y_train,
 y_cv) = train_test_split(X_train, y_train,
                          stratify=y_train, test_size=0.20, random_state=0)

print("The shape of X_train dataset: {}".format(X_train.shape))
print("The shape of X_cv dataset: {}".format(X_cv.shape))
print("The shape of X_test dataset: {}".format(X_test.shape))

print("The shape of y_train dataset: {}".format(y_train.shape))
print("The shape of y_cv dataset: {}".format(y_cv.shape))
print("The shape of y_test dataset: {}".format(y_test.shape))

# __4.2. Feature engineering__

# Feature engineering is the process of using domain knowledge to extract features from raw data via data mining techniques. These features can be used to improve the performance of machine learning algorithms. Feature engineering can be considered as applied machine learning itself [8].

# 4.2.1. Data normalization
# 
# The kernel density plots of the columns do not follow Gaussian distibution and moreover the columns are not having consistent scale. It is important to bring the values of all the columns into a consistent scale without distorting the meaning of the values. 

from sklearn.preprocessing import MinMaxScaler

scaling = MinMaxScaler()

X_train = scaling.fit_transform(X=X_train)
X_cv = scaling.transform(X=X_cv)
X_test = scaling.transform(X=X_test)

# 4.2.2. Values within the same band
# 
# The dataset has photometric filters - _u_, _g_, _r_, _i_, _z_ [6].
# 
# * Ultraviolet band: _u_.
#     * u: ultraviolet filter
# * Visible band: _g_, _r_.
#     * g: green filter
#     * r: red filter
# * Near-Infrared: _i_, _z_.
#     * i, z: infrared filters
# 
# I tried multiplying the values of the filters that belong to same category as mentioned above, but it did not yield better results. In the dataset, there is a column called _redshift_ which tallks about light stretching towards the red part of the spectrum. Hence I tried substracting the filter values of _r_ column. This improved the performance of the logistic regression model to some extent.
# 
# I also noted that columns - _alpha_ and _delta_ that imply to ascension and declination respectively do not contribute anything to model's performance.

def make_dataframe(arr, cols):
    """
    This function builts the dataframe.
    
    Parameters
    ----------
    `arr`: array of 2 dimension
    `cols`: column names
    
    Returns a dataframe.
    """
    df = pd.DataFrame(data=arr, columns=cols)
    return df

train_df_fea = make_dataframe(arr=X_train, cols=imp_cols)
cv_df_fea = make_dataframe(arr=X_cv, cols=imp_cols)
test_df_fea = make_dataframe(arr=X_test, cols=imp_cols)

def featurize(df):
    """
    This function featurizes the dataframe.
    
    Parameter
    ---------
    `df`: dataframe
    
    Returns a dataframe.
    """
    df['g-r'] = df['g'] - df['r']
    df['i-z'] = df['i'] - df['z']
    df['u-r'] = df['u'] - df['r']
    df['i-r'] = df['i'] - df['r']
    df['z-r'] = df['z'] - df['r']
    return df

train_df_fea = featurize(df=train_df_fea)
cv_df_fea = featurize(df=cv_df_fea)
test_df_fea = featurize(df=test_df_fea)

fea_cols = ['u', 'g', 'r', 'redshift', 'g-r', 'i-z', 'u-r', 'i-r', 'z-r']
print(fea_cols)

X_train_fea = train_df_fea[fea_cols].values
X_cv_fea = cv_df_fea[fea_cols].values
X_test_fea = test_df_fea[fea_cols].values

# 4.2.3. Principal component analysis (PCA)
# 
# PCA is a dimensionality reduction that identifies important relationships in the data, transforms the existing data based on these relationships, and then quantifies the importance of these relationships that can be used for modeling.

from sklearn.decomposition import PCA

cols_for_pca = ['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'redshift']

train_df_for_pca = train_df_fea[cols_for_pca]
cv_df_for_pca = cv_df_fea[cols_for_pca]
test_df_for_pca = test_df_fea[cols_for_pca]

pca_decomposer = PCA(n_components=0.95)

train_df_after_pca = pca_decomposer.fit_transform(X=train_df_for_pca)
cv_df_after_pca = pca_decomposer.transform(X=cv_df_for_pca)
test_df_after_pca = pca_decomposer.transform(X=test_df_for_pca)

pca_cols = ['f{}'.format(i+1) for i in range(train_df_after_pca.shape[1])]
print(pca_cols)

train_df_after_pca = make_dataframe(arr=train_df_after_pca, cols=pca_cols)
cv_df_after_pca = make_dataframe(arr=cv_df_after_pca, cols=pca_cols)
test_df_after_pca = make_dataframe(arr=test_df_after_pca, cols=pca_cols)

def scree_plotter(decomposer):
    """
    Draws the scree plot.
    
    Parameter
    ---------
    `decomposer`: pca object
    
    Returns none.
    """
    x_ = np.arange(pca_decomposer.n_components_)
    y = pca_decomposer.explained_variance_ratio_
    cum_y = np.cumsum(a=y)
    
    plt.figure(figsize=(7, 5))
    bars = sns.barplot(x=x_, y=cum_y)
    for b in bars.patches:
        x = b.get_x() + (b.get_width() / 2)
        y = np.round(b.get_height(), 3)
        bars.annotate(text=format(y),
                      xy=(x, y), ha='center', va='center', size=8, 
                      xytext=(0, 6), textcoords='offset points')
    plt.title('Scree Plot', fontsize=10)
    plt.xlabel('Principal Components', fontsize=9)
    plt.ylabel('Cumulative Variance Explained', fontsize=9)
    plt.xticks(ticks=x_)
    plt.show()

scree_plotter(decomposer=pca_decomposer)

X_train_pca = train_df_after_pca.values
X_cv_pca = cv_df_after_pca.values
X_test_pca = test_df_after_pca.values

# __4.3. Confusion, Precision, and Recall matrices__

from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             log_loss,
                             precision_score,
                             recall_score)

def plot_heatmap(matrix, title, labels):
    """
    This function plots the heatmap.
    
    Parameters
    ----------
    `matrix`: 2D array
    `title`: title
    `labels`: integer encoded target values
    
    Returns none.
    """
    sns.heatmap(data=matrix, annot=True, fmt='.2f', linewidths=0.1,
                xticklabels=labels, yticklabels=labels)
    plt.xlabel(xlabel='Predicted Class')
    plt.ylabel(ylabel='Actual Class')
    plt.title(label=title, fontsize=10)

def plot_confusion_matrix(y_true, y_pred, labels):
    """
    This function plots:
        1. Confusion matrix
        2. Precision matrix
        3. Recall matrix
    
    Parameters
    ----------
    `y_true`: ground truth (or actual) values
    `y_pred`: predicted values
    `labels`: integer encoded target values
    
    Returns none.
    """
    cmat = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
    pmat = cmat / cmat.sum(axis=0)
    print("Column sum of precision matrix: {}".format(pmat.sum(axis=0)))
    rmat = ((cmat.T) / (cmat.sum(axis=1).T)).T
    print("Row sum of recall matrix:       {}".format(rmat.sum(axis=1)))
    
    plt.figure(figsize=(15, 3))
    plt.subplot(131)
    plot_heatmap(matrix=cmat, title='Confusion Matrix', labels=labels)
    plt.subplot(132)
    plot_heatmap(matrix=pmat, title='Precision Matrix', labels=labels)
    plt.subplot(133)
    plot_heatmap(matrix=rmat, title='Recall Matrix', labels=labels)
    plt.show()

# __4.4. Logistic Regression__

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

def logistic_regresson(X_train,
                       X_cv,
                       X_test,
                       y_train,
                       y_cv,
                       y_test,
                       c_params,
                       cols,
                       model_name,
                       labels=labels):
    """
    This function builds the model.
    """
    if os.path.isdir('./model_dumps/eda_models'):
        pass
    else:
        os.mkdir(path='./model_dumps/eda_models')
    
    model_path = os.path.join('./model_dumps/eda_models', model_name)

    if not os.path.isfile(path=model_path):
        log_error_list = list()
        loss_df = pd.DataFrame()

        for c_i in c_params:
            clf = LogisticRegression(C=c_i, n_jobs=-1, random_state=42,
                                     max_iter=1000)
            clf.fit(X=X_train, y=y_train)

            sig_clf = CalibratedClassifierCV(base_estimator=clf)
            sig_clf.fit(X=X_train, y=y_train)

            cv_pred = sig_clf.predict_proba(X=X_cv)
            cv_l = log_loss(y_true=y_cv, y_pred=cv_pred)
            log_error_list.append(cv_l)

        print("Hyperparameter Tuning")
        loss_df['C'] = c_params
        loss_df['logloss'] = log_error_list
        loss_tb = tabulate(tabular_data=loss_df, headers='keys',
                           tablefmt='pretty')
        print(loss_tb)

        plt.figure(figsize=(6, 4))
        plt.plot(loss_df['C'], loss_df['logloss'], 'go--')
        for i, txt in enumerate(np.round(loss_df['logloss'].values, 3)):
            plt.annotate(text=(c_params[i], txt),
                         xy=(c_params[i], loss_df['logloss'].values[i]))
        plt.title(label='Cross Validation Error vs C')
        plt.xlabel(xlabel='C')
        plt.ylabel(ylabel='Error')
        plt.grid()
        plt.show()

        b_i = np.argmin(a=log_error_list)
        b_c = c_params[b_i]

        clf = LogisticRegression(n_jobs=-1, max_iter=1000, C=b_c,
                                 random_state=42)
        clf.fit(X=X_train, y=y_train)

        sig_clf = CalibratedClassifierCV(base_estimator=clf, method='sigmoid')
        sig_clf.fit(X=X_train, y=y_train)

        with open(file=model_path, mode='wb') as m_pkl:
            pickle.dump(obj=(clf, sig_clf, b_c), file=m_pkl)
        print("Model saved into the disk.")
    else:
        with open(file=model_path, mode='rb') as m_pkl:
            clf, sig_clf, b_c = pickle.load(file=m_pkl)
        print("Loaded the saved model from the disk.")
    
    train_pred = sig_clf.predict_proba(X=X_train)
    train_loss = log_loss(y_true=y_train, y_pred=train_pred, labels=labels)
    train_loss = np.round(train_loss, 3)
    train_cm_pred = sig_clf.predict(X=X_train)
    print("\nTrain")
    print("Logloss (Train): {} for the best C: {}".format(train_loss, b_c))
    plot_confusion_matrix(y_true=y_train, y_pred=train_cm_pred, labels=labels)
    print(classification_report(y_true=y_train, y_pred=train_cm_pred))
    
    cv_pred = sig_clf.predict_proba(X=X_cv)
    cv_loss = log_loss(y_true=y_cv, y_pred=cv_pred, labels=labels)
    cv_loss = np.round(cv_loss, 3)
    cv_cm_pred = sig_clf.predict(X=X_cv)
    print("Cross Validation")
    print("Logloss (CV): {} for the best C: {}".format(cv_loss, b_c))
    plot_confusion_matrix(y_true=y_cv, y_pred=cv_cm_pred, labels=labels)
    print(classification_report(y_true=y_cv, y_pred=cv_cm_pred))

    test_pred = sig_clf.predict_proba(X=X_test)
    test_loss = log_loss(y_true=y_test, y_pred=test_pred, labels=labels)
    test_loss = np.round(test_loss, 3)
    test_cm_pred = sig_clf.predict(X=X_test)
    print("Test")
    print("Logloss (Test): {} for the best C: {}".format(test_loss, b_c))
    plot_confusion_matrix(y_true=y_test, y_pred=test_cm_pred, labels=labels)
    print(classification_report(y_true=y_test, y_pred=test_cm_pred))

    feature_imp = clf.coef_

    plt.figure(figsize=(15, 3))
    for i, cls, fi in zip(range(len(labels)), labels, feature_imp):
        plt.subplot(int('13{}'.format(i+1)))
        plt.bar(x=cols, height=fi)
        plt.xticks(rotation=45)
        plt.title(label="Feature Importance: {}".format(cls),
                  fontsize=10)
        plt.grid()
    plt.show()

    return train_loss, cv_loss, test_loss

# __4.5. Testing the feature-engineered columns and PCA components on logistic regression__

# 4.5.1. Logistic regression on feature-engineered columns

model_name = 'logistic_regression_fea.pkl'

c_params = [10 ** x for x in range(-4, 1)]

(logreg_tr_loss,
 logreg_cv_loss,
 logreg_te_loss) = logistic_regresson(X_train=X_train_fea,
                                      y_train=y_train,
                                      X_cv=X_cv_fea,
                                      y_cv=y_cv,
                                      X_test=X_test_fea,
                                      y_test=y_test,
                                      c_params=c_params,
                                      cols=fea_cols,
                                      model_name=model_name)

# 4.5.2. Logistic regression on PCA components

model_name = 'logistic_regression_pca.pkl'

c_params = [10 ** x for x in range(-4, 1)]

(logreg_tr_loss_pca,
 logreg_cv_loss_pca,
 logreg_te_loss_pca) = logistic_regresson(X_train=X_train_pca,
                                          y_train=y_train,
                                          X_cv=X_cv_pca,
                                          y_cv=y_cv,
                                          X_test=X_test_pca,
                                          y_test=y_test,
                                          c_params=c_params,
                                          cols=pca_cols,
                                          model_name=model_name)

# 4.5.3. Logistic regression on feature-engineered columns and PCA components

train_df_fea_pca = pd.concat(objs=[train_df_fea, train_df_after_pca], axis=1)
cv_df_fea_pca = pd.concat(objs=[cv_df_fea, cv_df_after_pca], axis=1)
test_df_fea_pca = pd.concat(objs=[test_df_fea, test_df_after_pca], axis=1)

fea_pca_cols = fea_cols + pca_cols
print(fea_pca_cols)

X_train_fea_pca = train_df_fea_pca[fea_pca_cols].values
X_cv_fea_pca = cv_df_fea_pca[fea_pca_cols].values
X_test_fea_pca = test_df_fea_pca[fea_pca_cols].values

model_name = 'logistic_regression_fea_pca.pkl'

c_params = [10 ** x for x in range(-4, 1)]

(logreg_tr_loss_fea_pca,
 logreg_cv_loss_fea_pca,
 logreg_te_loss_fea_pca) = logistic_regresson(X_train=X_train_fea_pca,
                                              y_train=y_train,
                                              X_cv=X_cv_fea_pca,
                                              y_cv=y_cv,
                                              X_test=X_test_fea_pca,
                                              y_test=y_test,
                                              c_params=c_params,
                                              cols=fea_pca_cols,
                                              model_name=model_name)



# In this notebook, I will be implementing a series of models on feature-engineered data.


# __1. Packages__

import warnings
warnings.filterwarnings('ignore')

from IPython.display import display

from matplotlib import pyplot as plt
from matplotlib import style
style.use(style='seaborn-v0_8-deep')

from tabulate import tabulate

import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns


# __2. Data__

tr_fea_df = pd.read_csv(filepath_or_buffer='./data/train_fea.csv')
cv_fea_df = pd.read_csv(filepath_or_buffer='./data/cv_fea.csv')
te_fea_df = pd.read_csv(filepath_or_buffer='./data/test_fea.csv')

fea_cols = list(tr_fea_df.columns)
target = fea_cols.pop()
print(fea_cols)
print(target)

labels = cv_fea_df['class'].unique()
print(labels)

X_train = tr_fea_df[fea_cols].values
y_train = tr_fea_df[target].values

X_cv = cv_fea_df[fea_cols].values
y_cv = cv_fea_df[target].values

X_test = te_fea_df[fea_cols].values
y_test = te_fea_df[target].values

print(X_train.shape, y_train.shape)
print(X_cv.shape, y_cv.shape)
print(X_test.shape, y_test.shape)


# __3. Confusion, Precision, and Recall matrices__

from sklearn.metrics import confusion_matrix

def plot_heatmap(matrix, title, labels):
    """
    This function plots the heatmap.
    
    Parameters
    ----------
    `matrix`: 2D array
    `title`: title
    `labels`: target values
    
    Returns none.
    """
    sns.heatmap(data=matrix, annot=True, fmt='.2f', linewidths=0.1,
                xticklabels=labels, yticklabels=labels)
    plt.xlabel(xlabel='Predicted Class')
    plt.ylabel(ylabel='Actual Class')
    plt.title(label=title, fontsize=10)

def plot_confusion_matrix(y_true, y_pred, labels):
    """
    This function plots:
        1. Confusion matrix
        2. Precision matrix
        3. Recall matrix
    
    Parameters
    ----------
    `y_true`: ground truth (or actual) values
    `y_pred`: predicted values
    `labels`: target values
    
    Returns none.
    """
    cmat = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
    pmat = cmat / cmat.sum(axis=0)
    print("Column sum of precision matrix: {}".format(pmat.sum(axis=0)))
    rmat = ((cmat.T) / (cmat.sum(axis=1).T)).T
    print("Row sum of recall matrix:       {}".format(rmat.sum(axis=1)))
    
    plt.figure(figsize=(15, 3))
    plt.subplot(131)
    plot_heatmap(matrix=cmat, title='Confusion Matrix', labels=labels)
    plt.subplot(132)
    plot_heatmap(matrix=pmat, title='Precision Matrix', labels=labels)
    plt.subplot(133)
    plot_heatmap(matrix=rmat, title='Recall Matrix', labels=labels)
    plt.show()

from sklearn.metrics import classification_report
from sklearn.metrics import log_loss

def reporter(clf, X, y, title, labels, best=None):
    """
    This functions generates the report.
    
    Parameters
    ----------
    `clf`: classifier object
    `X`: features
    `y`: target
    `title`: title of the report
    `labels`: target values
    `best`: best parameters which are learned
    
    Returns logloss.
    """
    pred = clf.predict_proba(X=X)
    
    loss = log_loss(y_true=y, y_pred=pred)
    loss = np.round(a=loss, decimals=3)
    
    cm_pred = clf.predict(X=X)
    
    print(title)
    if best is None:
        print("Logloss: {}".format(loss))
    else:
        print("Logloss: {}".format(loss))
        print("Best parameters: {}".format(best))
    
    plot_confusion_matrix(y_true=y, y_pred=cm_pred, labels=labels)
    
    print(classification_report(y_true=y, y_pred=cm_pred))
    
    return loss


# __4. Modeling__


# 4.1. RandomizedSearchCV for hyperparameter tuning

from sklearn.model_selection import RandomizedSearchCV

def tuner(clf, dist, X, y):
    """
    This function tunes the hyperparameters.
    
    Parameters
    ----------
    `clf`: estimator object
    `dist`: hyperparameters to be tuned
    `X`: features
    `y`: target
    
    Returns the best values for hyperparameters.
    """
    rs_clf = RandomizedSearchCV(estimator=clf, random_state=0, n_jobs=-1,
                                param_distributions=dist)
    search = rs_clf.fit(X=X, y=y)
    return search.best_params_


# 4.2. Models

def get_model_path(model_name):
    """
    This function gets the model path.
    
    Parameter
    ---------
    `model_name`: name of the model
    
    Returns path of the model.
    """
    if os.path.isdir('./model_dumps'):
        pass
    else:
        os.mkdir(path='./model_dumps')
    
    model_path = os.path.join('./model_dumps', model_name)
    return model_path


# 4.2.1. Random or dummy model
# 
# The purpose of building a random or a dummy classifier is, the output of a multi-class logloss function is bounded between $0$ and $\infty$. Hence I will be using a dummy classifier to benchmark the performance of the worst model. Any real model that I will build should perform better than a random or a dummy classifier.

from sklearn.dummy import DummyClassifier

def dummy_classifier(X_train,
                     y_train,
                     X_cv,
                     y_cv,
                     X_test,
                     y_test,
                     model_name,
                     labels=labels):
    """
    A random or dummy model.
    """
    model_path = get_model_path(model_name=model_name)
    
    if not os.path.isfile(path=model_path):
        clf = DummyClassifier(strategy='uniform')
        clf.fit(X=X_train, y=y_train)
        
        with open(file=model_path, mode='wb') as m_pkl:
            pickle.dump(obj=clf, file=m_pkl)
        print("Model saved into the disk.\n")
    else:
        with open(file=model_path, mode='rb') as m_pkl:
            clf = pickle.load(file=m_pkl)
        print("Loaded the saved model from the disk.\n")
    
    tr_loss = reporter(clf=clf, X=X_train, y=y_train,
                       title='Train', labels=labels)
    cv_loss = reporter(clf=clf, X=X_cv, y=y_cv,
                       title='Cross Validation', labels=labels)
    te_loss = reporter(clf=clf, X=X_test, y=y_test,
                       title='Test', labels=labels)
    
    return tr_loss, cv_loss, te_loss

model_name = 'model_dummy_classifier.pkl'

(dummy_tr_loss,
 dummy_cv_loss,
 dummy_te_loss) = dummy_classifier(X_train=X_train,
                                   y_train=y_train,
                                   X_cv=X_cv,
                                   y_cv=y_cv,
                                   X_test=X_test,
                                   y_test=y_test,
                                   model_name=model_name)


# 4.2.2. Logistic regression

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

def logistic_regresson(X_train,
                       y_train,
                       X_cv,
                       y_cv,
                       X_test,
                       y_test,
                       dist,
                       model_name,
                       labels=labels):
    """
    This function builds the logistic regression model.
    """
    model_path = get_model_path(model_name=model_name)

    if not os.path.isfile(path=model_path):
        clf = LogisticRegression(n_jobs=-1, random_state=42, max_iter=1000, 
                                 class_weight='balanced')

        best = tuner(clf=clf, dist=dist, X=X_train, y=y_train)

        clf = LogisticRegression(n_jobs=-1, max_iter=1000, C=best['C'],
                                 random_state=42, penalty=best['penalty'],
                                 class_weight='balanced')
        clf.fit(X=X_train, y=y_train)

        sig_clf = CalibratedClassifierCV(base_estimator=clf)
        sig_clf.fit(X=X_train, y=y_train)

        with open(file=model_path, mode='wb') as m_pkl:
            pickle.dump(obj=(clf, sig_clf, best), file=m_pkl)
        print("Model saved into the disk.\n")
    else:
        with open(file=model_path, mode='rb') as m_pkl:
            clf, sig_clf, best = pickle.load(file=m_pkl)
        print("Loaded the saved model from the disk.\n")
    
    tr_loss = reporter(clf=sig_clf, X=X_train, y=y_train,
                       title='Train', best=best, labels=labels)
    cv_loss = reporter(clf=sig_clf, X=X_cv, y=y_cv,
                       title='Cross Validation', best=best, labels=labels)
    te_loss = reporter(clf=sig_clf, X=X_test, y=y_test,
                       title='Test', best=best, labels=labels)
    
    return best, tr_loss, cv_loss, te_loss

model_name = 'model_logistic_regression.pkl'

dist = dict(C=[10 ** x for x in range(-4, 3)], penalty=['l2', 'l1'])

(logreg_best,
 logreg_tr_loss,
 logreg_cv_loss,
 logreg_te_loss) = logistic_regresson(X_train=X_train,
                                      y_train=y_train,
                                      X_cv=X_cv,
                                      y_cv=y_cv,
                                      X_test=X_test,
                                      y_test=y_test,
                                      dist=dist,
                                      model_name=model_name)


# 4.2.3. Support vector classifier

from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

def support_vector_classifier(X_train,
                              y_train,
                              X_cv,
                              y_cv,
                              X_test,
                              y_test,
                              dist,
                              model_name,
                              labels=labels):
    """
    This function builds the support vector classifier model.
    """
    model_path = get_model_path(model_name=model_name)

    if not os.path.isfile(path=model_path):
        clf = SVC(random_state=42, class_weight='balanced')

        best = tuner(clf=clf, dist=dist, X=X_train, y=y_train)

        clf = SVC(C=best['C'], random_state=42, class_weight='balanced')
        clf.fit(X=X_train, y=y_train)

        sig_clf = CalibratedClassifierCV(base_estimator=clf)
        sig_clf.fit(X=X_train, y=y_train)

        with open(file=model_path, mode='wb') as m_pkl:
            pickle.dump(obj=(clf, sig_clf, best), file=m_pkl)
        print("Model saved into the disk.\n")
    else:
        with open(file=model_path, mode='rb') as m_pkl:
            clf, sig_clf, best = pickle.load(file=m_pkl)
        print("Loaded the saved model from the disk.\n")
    
    tr_loss = reporter(clf=sig_clf, X=X_train, y=y_train,
                       title='Train', best=best, labels=labels)
    cv_loss = reporter(clf=sig_clf, X=X_cv, y=y_cv,
                       title='Cross Validation', best=best, labels=labels)
    te_loss = reporter(clf=sig_clf, X=X_test, y=y_test,
                       title='Test', best=best, labels=labels)
    
    return best, tr_loss, cv_loss, te_loss

model_name = 'model_support_vector_classifier.pkl'

dist = dict(C=[10 ** x for x in range(-4, 3)])

(svc_best,
 svc_tr_loss,
 svc_cv_loss,
 svc_te_loss) = support_vector_classifier(X_train=X_train,
                                          y_train=y_train,
                                          X_cv=X_cv,
                                          y_cv=y_cv,
                                          X_test=X_test,
                                          y_test=y_test,
                                          dist=dist,
                                          model_name=model_name)


# 4.2.4. K neighbors classifier

from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier

def k_neighbors_classifier(X_train,
                           y_train,
                           X_cv,
                           y_cv,
                           X_test,
                           y_test,
                           dist,
                           model_name,
                           labels=labels):
    """
    This function builds the k neighbors classifier.
    """
    model_path = get_model_path(model_name=model_name)

    if not os.path.isfile(path=model_path):
        clf = KNeighborsClassifier(n_jobs=-1)

        best = tuner(clf=clf, dist=dist, X=X_train, y=y_train)

        clf = KNeighborsClassifier(n_jobs=-1, n_neighbors=best['n_neighbors'])
        clf.fit(X=X_train, y=y_train)

        sig_clf = CalibratedClassifierCV(base_estimator=clf)
        sig_clf.fit(X=X_train, y=y_train)

        with open(file=model_path, mode='wb') as m_pkl:
            pickle.dump(obj=(clf, sig_clf, best), file=m_pkl)
        print("Model saved into the disk.\n")
    else:
        with open(file=model_path, mode='rb') as m_pkl:
            clf, sig_clf, best = pickle.load(file=m_pkl)
        print("Loaded the saved model from the disk.\n")
    
    tr_loss = reporter(clf=sig_clf, X=X_train, y=y_train,
                       title='Train', best=best, labels=labels)
    cv_loss = reporter(clf=sig_clf, X=X_cv, y=y_cv,
                       title='Cross Validation', best=best, labels=labels)
    te_loss = reporter(clf=sig_clf, X=X_test, y=y_test,
                       title='Test', best=best, labels=labels)
    
    return best, tr_loss, cv_loss, te_loss

model_name = 'model_k_neighbors_classifier.pkl'

dist = dict(n_neighbors=[3, 5, 11, 15, 21, 31, 41, 51, 99])

(knn_best,
 knn_tr_loss,
 knn_cv_loss,
 knn_te_loss) = k_neighbors_classifier(X_train=X_train,
                                       y_train=y_train,
                                       X_cv=X_cv,
                                       y_cv=y_cv,
                                       X_test=X_test,
                                       y_test=y_test,
                                       dist=dist,
                                       model_name=model_name)


# 4.2.5. Decision tree classifier

from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier

def decision_tree_classifier(X_train,
                             y_train,
                             X_cv,
                             y_cv,
                             X_test,
                             y_test,
                             dist,
                             model_name,
                             labels=labels):
    """
    This function builds the decision tree classifier.
    """
    model_path = get_model_path(model_name=model_name)

    if not os.path.isfile(path=model_path):
        clf = DecisionTreeClassifier(random_state=42)

        best = tuner(clf=clf, dist=dist, X=X_train, y=y_train)

        clf = DecisionTreeClassifier(criterion=best['criterion'],
                                     max_depth=best['max_depth'],
                                     min_samples_split=best['min_samples_split'],
                                     random_state=42)
        clf.fit(X=X_train, y=y_train)

        sig_clf = CalibratedClassifierCV(base_estimator=clf)
        sig_clf.fit(X=X_train, y=y_train)

        with open(file=model_path, mode='wb') as m_pkl:
            pickle.dump(obj=(clf, sig_clf, best), file=m_pkl)
        print("Model saved into the disk.\n")
    else:
        with open(file=model_path, mode='rb') as m_pkl:
            clf, sig_clf, best = pickle.load(file=m_pkl)
        print("Loaded the saved model from the disk.\n")
    
    tr_loss = reporter(clf=sig_clf, X=X_train, y=y_train,
                       title='Train', best=best, labels=labels)
    cv_loss = reporter(clf=sig_clf, X=X_cv, y=y_cv,
                       title='Cross Validation', best=best, labels=labels)
    te_loss = reporter(clf=sig_clf, X=X_test, y=y_test,
                       title='Test', best=best, labels=labels)
    
    return best, tr_loss, cv_loss, te_loss

model_name = 'model_decision_tree_classifier.pkl'

dist = dict(criterion=['gini', 'entropy', 'log_loss'],
            max_depth=[1, 5, 10, 50, 100],
            min_samples_split=[5, 10, 100, 250, 500])

(dt_best,
 dt_tr_loss,
 dt_cv_loss,
 dt_te_loss) = decision_tree_classifier(X_train=X_train,
                                        y_train=y_train,
                                        X_cv=X_cv,
                                        y_cv=y_cv,
                                        X_test=X_test,
                                        y_test=y_test,
                                        dist=dist,
                                        model_name=model_name)


# 4.2.6. Random forest classifier

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

def feature_importance_plot(data, x, y, title):
    """
    This function plots the feature importance plot.
    
    Parameters
    ----------
    `data`: dataframe
    `x`: x-axis
    `y`: y-axis
    `title`: title of the plot
    
    Returns none.
    """
    bars = sns.barplot(data=data, x=x, y=y)
    for b in bars.patches:
        x = b.get_x() + (b.get_width() / 2)
        y = np.round(b.get_height(), 3)
        bars.annotate(text=format(y),
                      xy=(x, y), ha='center', va='center', size=8, 
                      xytext=(0, 6), textcoords='offset points')
    plt.title(label=title)

def random_forest_classifier(X_train,
                             y_train,
                             X_cv,
                             y_cv,
                             X_test,
                             y_test,
                             dist,
                             model_name,
                             labels=labels,
                             plot_fi=False):
    """
    This function builds the random forest classifier.
    """
    model_path = get_model_path(model_name=model_name)

    if not os.path.isfile(path=model_path):
        clf = RandomForestClassifier(n_jobs=-1, random_state=42)

        best = tuner(clf=clf, dist=dist, X=X_train, y=y_train)

        clf = RandomForestClassifier(n_estimators=best['n_estimators'],
                                     criterion=best['criterion'],
                                     max_depth=best['max_depth'],
                                     min_samples_split=best['min_samples_split'],
                                     n_jobs=-1, random_state=42)
        clf.fit(X=X_train, y=y_train)

        sig_clf = CalibratedClassifierCV(base_estimator=clf)
        sig_clf.fit(X=X_train, y=y_train)

        with open(file=model_path, mode='wb') as m_pkl:
            pickle.dump(obj=(clf, sig_clf, best), file=m_pkl)
        print("Model saved into the disk.\n")
    else:
        with open(file=model_path, mode='rb') as m_pkl:
            clf, sig_clf, best = pickle.load(file=m_pkl)
        print("Loaded the saved model from the disk.\n")
    
    if plot_fi:
        imp_df = pd.DataFrame()
        imp_df['Features'] = fea_cols
        imp_df['Importance'] = clf.feature_importances_
        imp_df = imp_df.sort_values(by=['Importance'], ascending=False)
        imp_df['Cumulative'] = np.cumsum(a=imp_df['Importance'].values)
    
        plt.figure(figsize=(15, 5))
        plt.subplot(121)
        feature_importance_plot(data=imp_df, x='Features', y='Importance',
                                title='Feature Importace')
        plt.subplot(122)
        feature_importance_plot(data=imp_df, x='Features', y='Cumulative',
                                title='Cumulative Feature Importance')
        plt.show()
    else:
        pass
    
    tr_loss = reporter(clf=sig_clf, X=X_train, y=y_train,
                       title='Train', best=best, labels=labels)
    cv_loss = reporter(clf=sig_clf, X=X_cv, y=y_cv,
                       title='Cross Validation', best=best, labels=labels)
    te_loss = reporter(clf=sig_clf, X=X_test, y=y_test,
                       title='Test', best=best, labels=labels)
    
    return best, tr_loss, cv_loss, te_loss

model_name = 'model_random_forest_classifier.pkl'

dist = dict(n_estimators=[5, 10, 25, 50, 100, 200, 250, 500, 1000],
            criterion=['gini', 'entropy', 'log_loss'],
            max_depth=[1, 5, 10, 25, 50, 100, 150],
            min_samples_split=[1, 5, 10, 25, 50, 100, 250, 500])

(rf_best,
 rf_tr_loss,
 rf_cv_loss,
 rf_te_loss) = random_forest_classifier(X_train=X_train,
                                        y_train=y_train,
                                        X_cv=X_cv,
                                        y_cv=y_cv,
                                        X_test=X_test,
                                        y_test=y_test,
                                        dist=dist,
                                        model_name=model_name,
                                        plot_fi=True)


# 4.2.6.1. Selecting important features using random forests


# From feature importance plot from the above cell, I am taking features that contribute atmost $96\%$ of importance. The features that contribute $96\%$ of importance are - _g_, _redshift_, _g-r_, _i-z_, _u-r_, _i-r_, _z-r_.

fi_cols = ['redshift', 'g-r', 'i-z', 'u-r', 'i-r', 'z-r', 'g']

fi_tr_data = tr_fea_df[fi_cols]
fi_cv_data = cv_fea_df[fi_cols]
fi_te_data = te_fea_df[fi_cols]

def export_data(data, target_arr, filename):
    """
    This function exports the data.
    
    Parameters
    ----------
    `data`: dataframe
    `filename`: the filename that data will be exported to
    """
    if os.path.isdir('./data/fi_data'):
        pass
    else:
        os.mkdir(path='./data/fi_data')
    
    data['class'] = target_arr
    data.to_csv(path_or_buf=os.path.join('./data/fi_data', filename), index=None)
    print("The data is exported to '{}'.".format(filename))

export_data(data=fi_tr_data, target_arr=y_train, filename='fi_tr_data.csv')

export_data(data=fi_cv_data, target_arr=y_cv, filename='fi_cv_data.csv')

export_data(data=fi_te_data, target_arr=y_test, filename='fi_te_data.csv')


# 4.2.7. XGBoost classifier

from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

def encode_targets(y_train, y_cv, y_test, labels):
    """
    This function encodes the targets.
    
    Parameters
    ----------
    `y_train`: targets in train set
    `y_cv`: targets in cross validation set
    `y_test`: targets in test set
    `labels`: target values
    
    Returns a tuple of encoded target sets.
    """
    encoder = LabelEncoder()
    
    y_train = encoder.fit_transform(y=y_train)
    y_cv = encoder.transform(y=y_cv)
    y_test = encoder.transform(y=y_test)
    
    labels = encoder.transform(y=labels)
    
    return y_train, y_cv, y_test, labels

def xgb_classifier(X_train,
                   y_train,
                   X_cv,
                   y_cv,
                   X_test,
                   y_test,
                   dist,
                   model_name,
                   labels=labels):
    """
    This function builds the xgb classifier.
    """
    model_path = get_model_path(model_name=model_name)
    
    (y_train, y_cv,
     y_test, labels) = encode_targets(y_train=y_train, y_cv=y_cv,
                                      y_test=y_test, labels=labels)

    if not os.path.isfile(path=model_path):
        clf = XGBClassifier(n_jobs=-1, random_state=42)

        best = tuner(clf=clf, dist=dist, X=X_train, y=y_train)

        clf = XGBClassifier(n_estimators=best['n_estimators'],
                            max_depth=best['max_depth'],
                            n_jobs=-1, random_state=42)
        clf.fit(X=X_train, y=y_train)

        sig_clf = CalibratedClassifierCV(base_estimator=clf)
        sig_clf.fit(X=X_train, y=y_train)

        with open(file=model_path, mode='wb') as m_pkl:
            pickle.dump(obj=(clf, sig_clf, best), file=m_pkl)
        print("Model saved into the disk.\n")
    else:
        with open(file=model_path, mode='rb') as m_pkl:
            clf, sig_clf, best = pickle.load(file=m_pkl)
        print("Loaded the saved model from the disk.\n")
    
    tr_loss = reporter(clf=sig_clf, X=X_train, y=y_train,
                       title='Train', best=best, labels=labels)
    cv_loss = reporter(clf=sig_clf, X=X_cv, y=y_cv,
                       title='Cross Validation', best=best, labels=labels)
    te_loss = reporter(clf=sig_clf, X=X_test, y=y_test,
                       title='Test', best=best, labels=labels)
    
    return best, tr_loss, cv_loss, te_loss

model_name = 'model_xgb_classifier.pkl'

dist = dict(n_estimators=[5, 10, 25, 50, 100, 200, 250, 500, 1000],
            max_depth=[1, 5, 10, 25, 50, 100, 150])

(xgbc_best,
 xgbc_tr_loss,
 xgbc_cv_loss,
 xgbc_te_loss) = xgb_classifier(X_train=X_train,
                                y_train=y_train,
                                X_cv=X_cv,
                                y_cv=y_cv,
                                X_test=X_test,
                                y_test=y_test,
                                dist=dist,
                                model_name=model_name)


# 4.2.8. Stacking classifier


# Reference: [https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/](https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/)

from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

def evaluate_model(model, X, y):
    """
    Model evaluation for base learners in stacking classifier.
    
    Parameters
    ----------
    `model`: estimator
    `X`: features
    `y`: targets
    
    Returns cross validation scores.
    """
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(estimator=model,
                             X=X, y=y, scoring='accuracy',
                             cv=cv, n_jobs=-1,
                             error_score='raise')
    return scores

def stacking_classifier(X_train,
                        y_train,
                        X_cv,
                        y_cv,
                        X_test,
                        y_test,
                        models,
                        model_name,
                        labels=labels):
    """
    This function builds the stacking classifier.
    """
    model_path = get_model_path(model_name=model_name)
    
    if not os.path.isfile(path=model_path):
        clf = StackingClassifier(estimators=models)
        clf.fit(X=X_train, y=y_train)

        with open(file=model_path, mode='wb') as m_pkl:
            pickle.dump(obj=clf, file=m_pkl)
        print("Model saved into the disk.\n")
    else:
        with open(file=model_path, mode='rb') as m_pkl:
            clf = pickle.load(file=m_pkl)
        print("Loaded the saved model from the disk.\n")
    
    tr_loss = reporter(clf=clf, X=X_train, y=y_train,
                       title='Train', labels=labels)
    cv_loss = reporter(clf=clf, X=X_cv, y=y_cv,
                       title='Cross Validation', labels=labels)
    te_loss = reporter(clf=clf, X=X_test, y=y_test,
                       title='Test', labels=labels)
    
    return tr_loss, cv_loss, te_loss

model_name = 'model_stacking_classifier.pkl'

LR = LogisticRegression(penalty=logreg_best['penalty'],
                        C=logreg_best['C'],
                        class_weight='balanced',
                        random_state=42,
                        n_jobs=-1, max_iter=1000)

SV = SVC(C=svc_best['C'], random_state=42, class_weight='balanced')

KNN = KNeighborsClassifier(n_neighbors=knn_best['n_neighbors'], n_jobs=-1)

DT = DecisionTreeClassifier(criterion=dt_best['criterion'],
                            max_depth=dt_best['max_depth'],
                            min_samples_split=dt_best['min_samples_split'],
                            random_state=42)

models = [('LR', LR), ('SVC', SV), ('KNN', KNN), ('DT', DT)]

(stack_tr_loss,
 stack_cv_loss,
 stack_te_loss) = stacking_classifier(X_train=X_train,
                                      y_train=y_train,
                                      X_cv=X_cv,
                                      y_cv=y_cv,
                                      X_test=X_test,
                                      y_test=y_test,
                                      models=models,
                                      model_name=model_name)


# __5. Summary__

model_names = ['Dummy', 'Logistic Regression', 'Support Vector',
               'K-Nearest Neighbors', 'Decision Tree',
               'Random Forest', 'XGBoost', 'Stacking']

tr_losses = [dummy_tr_loss, logreg_tr_loss, svc_tr_loss, knn_tr_loss,
             dt_tr_loss, rf_tr_loss, xgbc_tr_loss, stack_tr_loss]
cv_losses = [dummy_cv_loss, logreg_cv_loss, svc_cv_loss, knn_cv_loss,
             dt_cv_loss, rf_cv_loss, xgbc_cv_loss, stack_cv_loss]
te_losses = [dummy_te_loss, logreg_te_loss, svc_te_loss, knn_te_loss,
             dt_te_loss, rf_te_loss, xgbc_te_loss, stack_te_loss]

summary_df = pd.DataFrame()
summary_df['Models'] = model_names
summary_df['Train Loss'] = tr_losses
summary_df['CV Loss'] = cv_losses
summary_df['Test Loss'] = te_losses

summary = tabulate(tabular_data=summary_df, headers='keys',
                   tablefmt='psql')
print(summary)

tidy = summary_df.melt(id_vars='Models').rename(columns=str.title)

plt.figure(figsize=(8, 4))
sns.barplot(data=tidy, x='Models', y='Value', hue='Variable', alpha=0.9)
plt.title(label='Logloss Obtained')
plt.xticks(rotation=90)
plt.show()


import warnings
warnings.filterwarnings('ignore')

from IPython.display import display

import numpy as np
import os
import pandas as pd
import pickle


# __2. Features and target__

features = ['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'redshift']
target = 'class'


# __3. Fetch the raw data__

def fetch_data(features):
    """
    This function fetches the raw data.
    """
    data = {f: [float(input("  '{}': ".format(f)))] for f in features}
    df = pd.DataFrame(data=data)
    print("Raw data is fetched successfully.")
    return df

df = fetch_data(features=features)


# __4. Preprocess the raw data__

def preprocess(df, features):
    """
    This function preprocess the rae data.
    """
    scale = 'analysis_dumps/scaling.pkl'
    with open(file=scale, mode='rb') as pre_pkl:
        scaling = pickle.load(file=pre_pkl)
    
    df = scaling.transform(X=df)
    df = pd.DataFrame(data=df, columns=features)
    return df

df = preprocess(df=df, features=features)
display(df)


# __5. Feature engineering on preprocessed data__

def featurize(df):
    """
    This function featurizes the dataframe.
    It selects the important features obtained using RF.
    Please refer 02-Modeling and 03-Modeling-FI notebooks.
    """
    fi_cols = ['redshift', 'g-r', 'i-z', 'u-r', 'i-r', 'z-r', 'g']
    df['g-r'] = df['g'] - df['r']
    df['i-z'] = df['i'] - df['z']
    df['u-r'] = df['u'] - df['r']
    df['i-r'] = df['i'] - df['r']
    df['z-r'] = df['z'] - df['r']
    df = df[fi_cols]
    return df

df = featurize(df=df)
display(df)


# __6. Predictions__

def prediction(X):
    """
    This functions predicts the datapoint.
    """
    model = 'model_dumps/fi_models/fi_model_stacking_classifier.pkl'
    with open(file=model, mode='rb') as m_pkl:
        clf = pickle.load(file=m_pkl)
    
    pred_proba = clf.predict_proba(X=X)
    confidence = np.round(a=np.max(pred_proba)*100, decimals=2)
    pred_class = clf.predict(X=X)[0]
    if pred_class == 'QSO': pred_class = 'Quasi-Stellar Object'
    elif pred_class == 'GALAXY': pred_class = 'Galaxy'
    else: pred_class = 'Star'
    print("The predicted class is '{}' with a confidence of {}%.".format(pred_class, confidence))

prediction(X=df)


# __7. Machine learning pipeline__


# For a single query point.

def ml_pipeline(features):
    """
    This is a local machine learning application.
    """
    print("Please provide the data for below features.")
    df = fetch_data(features=features)
    df = preprocess(df=df, features=features)
    df = featurize(df=df)
    prediction(X=df)

ml_pipeline(features=features)


# For the test data.

from sklearn.metrics import classification_report

def pipeline_for_whole_test_data(features, target='class'):
    """
    This function a pipeline for whole dataset.
    """
    data = pd.read_csv(filepath_or_buffer='data/test_data.csv')
    
    X_test = data[features]
    y_test = data[target].values
    
    X_test = featurize(df=X_test)
    
    model = 'model_dumps/fi_models/fi_model_stacking_classifier.pkl'
    with open(file=model, mode='rb') as m_pkl:
        clf = pickle.load(file=m_pkl)
    
    cm_pred = clf.predict(X=X_test)
    
    print(classification_report(y_true=y_test, y_pred=cm_pred))

pipeline_for_whole_test_data(features=features)



#clustering import pandas as pd
import numpy as np
from astropy.cosmology import Planck18 as cosmo
from astropy.coordinates import SkyCoord
from astropy import units as u
from sklearn.cluster import DBSCAN

# Assuming Astrodf is your DataFrame with columns 'alpha', 'delta', 'redshift'
# Example: Astrodf = pd.read_csv('your_dataset.csv')
Astrodf = data_df['alpha', 'delta', 'redshift']
# Step 2: Convert to 3D spatial coordinates
def convert_to_3d(alpha, delta, redshift):
    sky_coord = SkyCoord(ra=alpha*u.degree, dec=delta*u.degree, frame='icrs')
    dist = cosmo.luminosity_distance(redshift).to(u.Mpc).value
    x = dist * np.cos(sky_coord.ra.rad) * np.cos(sky_coord.dec.rad)
    y = dist * np.sin(sky_coord.ra.rad) * np.cos(sky_coord.dec.rad)
    z = dist * np.sin(sky_coord.dec.rad)
    return x, y, z

# Apply the conversion to the entire dataset
coords = np.array([convert_to_3d(row.alpha, row.delta, row.redshift) for index, row in Astrodf.iterrows()])

# Step 3: Clustering analysis
# Adjust these parameters based on your dataset
eps = 10  # Example value, in Mpc
min_samples = 10  # Minimum samples to form a cluster

clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(coords)

# Add cluster labels to your DataFrame
Astrodf['cluster'] = clustering.labels_

# Step 4: Interpretation and Validation
# This is more open-ended and will depend on your specific goals and dataset
