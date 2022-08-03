# %%
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.model_selection import StratifiedKFold
from copy import deepcopy
import matplotlib.pyplot as plt
# classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


datasetPahts = ['../dataset/tfidf_extracted.csv',
                '../dataset/doc2vec_extracted.csv']

# classifier names
names = [
    "5 Nearest Neighbors",
    "RBF SVM",
    "Logistic Regression",
    "Random Forest",
    "Naive Bayes",
]

# actual classifier
classifiers = [
    KNeighborsClassifier(5),
    SVC(kernel='rbf', probability=True),
    LogisticRegression(),
    RandomForestClassifier(),
    GaussianNB()
]

BinaryClassification = True

print("Loading Dataset ...")
dataset = pd.read_csv(datasetPahts[0], index_col=False)
if (BinaryClassification):
    dataset = dataset[dataset.Score != 0]
dataset = dataset.sample(10000)
print(dataset.shape)
print(dataset["Score"].value_counts())
print("loading complete!\n")


def splitXY(df):
    '''
    splits the dataset to X and Y
    '''
    x = df.loc[:, df.columns != 'Score']
    y = df.loc[:, df.columns == 'Score']
    return x, y


# for dataset in datasets:

# %%
X, Y = splitXY(dataset)
Y = Y['Score']


def makeBinary(data, posclass):
    '''
    sets positive class to 1 ,and the rest to 0
    for computing roc curve in multiclass one vs all
    '''
    bin = pd.DataFrame(data, columns=['Score'], copy=True)

    # separation because 1 and 0 already are in the dataset
    bin.loc[bin.Score == posclass] = 10
    bin.loc[bin.Score != 10] = -10

    # then
    bin.loc[bin.Score == 10] = 1
    bin.loc[bin.Score != 1] = 0

    return bin


splits = 10
cv = StratifiedKFold(n_splits=splits)

for classifier, name in zip(classifiers, names):
    print('Start: '+name)
    counter = 1
    acc = []
    f1 = []
    roc_auc = []
    roc = []
    mean_fpr = []
    mean_tpr = []
    # determines that mean_tpr and mean_ftr are empty
    flag = 0
    # stratiffied 10 fold cross validation
    for train, test in cv.split(X, Y):
        model = deepcopy(classifier)
        x_train = X.iloc[train]
        y_train = Y.iloc[train]
        x_test = X.iloc[test]
        y_test = Y.iloc[test]
        # train
        model.fit(x_train, y_train)
        # predict
        y_pred = model.predict(x_test)
        # y_proba = model.predict_proba(x_test)
        # metrics
        acc.append(accuracy_score(y_test, y_pred))
        # Roc Curve for each class and then mean
        # fpr = dict()
        # tpr = dict()
        # for positiveClass in range(-1, 2):
        #     bin_pred = makeBinary(y_pred, positiveClass)
        #     bin_true = makeBinary(y_test, positiveClass)
        #     # print(bin_pred, bin_true)
        #     fpr[positiveClass], tpr[positiveClass], _ = metrics.roc_curve(
        #         y_true=y_test, y_score=y_proba[:, positiveClass])
        #     # print(len(fpr), len(tpr))
        #     # if flag == 0:
        #     #     mean_fpr = fpr
        #     #     mean_tpr = tpr
        #     #     flag = 1
        #     # else:
        #     #     mean_fpr += fpr
        #     #     mean_tpr += tpr
        # all_fpr = np.unique(np.concatenate([fpr[i] for i in range(-1, 2)]))
        # mean_tpr = np.zeros_like(all_fpr)
        # for i in range(-1, 2):
        #     mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        # mean_tpr /= 3
        # plt.plot(all_fpr, mean_tpr, label=name)
        # roc_auc.append(auc(all_fpr, mean_tpr))
        if BinaryClassification:
            f1.append(f1_score(y_test, y_pred))
            roc_auc.append(roc_auc_score(y_test, y_pred))
        else:
            f1.append(f1_score(y_test, y_pred, average="weighted"))
            roc_auc.append(roc_auc_score(
                y_test, model.predict_proba(x_test), average="weighted", multi_class='ovr'))

        # print("Fold: "+str(counter)+",  Accuracy: "+str(sum(acc)/counter)+"   F1: " +
        #       str(sum(f1)/counter)+" AUC: "+str(sum(roc_auc)/counter))
        counter += 1

    print("Accuracy: "+str(sum(acc)/splits)+"   F1: " +
          str(sum(f1)/splits)+" AUC: "+str(sum(roc_auc)/splits))
# plt.legend()
# plt.show()


# %%
