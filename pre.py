import pandas as pd
import numpy as np
import os
import random
import sklearn
from sklearn.impute import SimpleImputer
import seaborn as sb
from scipy import stats
from sklearn.ensemble import IsolationForest
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.datasets import dump_svmlight_file
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectKBest, f_regression

if __name__ == '__main__':

    X = pd.read_csv(os.path.join("data", "X_train.csv"),
                    delimiter=',',
                    index_col = 'id')
    y = pd.read_csv(os.path.join("data", "y_train.csv"),
                    delimiter=',',
                    index_col='id')

    nan_idx = np.isnan(X)
    imputer = KNNImputer(n_neighbors=20)
    X_imp = imputer.fit_transform(X)

    model = sklearn.ensemble.IsolationForest(max_samples=200,
                                             contamination=0.025,
                                             max_features=len(X_imp[0]))
    model.fit(X_imp)
    outl_pred = model.predict(X_imp)
    mask = outl_pred != -1


    X_cleaned = np.copy(X_imp)
    y_cleaned = np.copy(y.values)
    X_cleaned, y_cleaned = X_cleaned[mask, :], y_cleaned[mask]
    print(X_cleaned.shape, y_cleaned.shape)

    """
    X_cleaned_std = sklearn.preprocessing.StandardScaler().fit(X_cleaned).transform(X_cleaned)

    df = pd.concat([pd.DataFrame(X_cleaned_std, columns=X.columns),
                    pd.DataFrame(y_cleaned, columns=y.columns)], axis=1)
    # df = pd.concat([pd.DataFrame(X_cleaned_std),pd.DataFrame(y_cleaned)],axis=1)
    df.rename(columns={df.columns[-1]: "y"}, inplace=True)

    corr = np.abs(df.corrwith(df["y"], method='spearman'))
    corrcoef = corr < 0.2

    cols = np.where(corr < 0.2)
    columnnames = np.array(X.columns)
    columnnames = columnnames[cols]

    df_new = df.drop(columns=columnnames)
    X_train = df_new.drop(columns='y')
    y_train = df_new['y']
    """

    X_train = X_cleaned
    y_train = y_cleaned
    selector = SelectKBest(score_func=f_regression, k=100)

    selector.fit(X_train, y_train)

    X_train = sklearn.preprocessing.StandardScaler().fit(X_cleaned).transform(X_train)
    X_train = selector.transform(X_train)

    X_test = pd.read_csv(os.path.join("data", "X_test.csv"),
                         delimiter=',',
                         index_col='id')

    X_test_KNN = imputer.fit(X_cleaned).transform(X_test)
    X_test = pd.DataFrame(X_test_KNN, columns=X_test.columns, index=X_test.index)

    X_test_stdised = sklearn.preprocessing.StandardScaler().fit(X_cleaned).transform(X_test)
    # X_test_stdised = np.delete(X_test_stdised, cols, axis=1)
    X_test_stdised = selector.transform(X_test_stdised)

    """
    n_train = int(len(X_train) * .8)
    ftrain_svm = open(os.path.join('data', 'train.svm.txt'), 'wb')
    dump_svmlight_file(X_train.to_numpy()[:n_train], y_train.to_numpy()[:n_train], ftrain_svm)
    ftrain_svm.close()

    fval_svm = open(os.path.join('data', 'val.svm.txt'), 'wb')
    dump_svmlight_file(X_train.to_numpy()[n_train:], y_train.to_numpy()[n_train:], fval_svm)
    fval_svm.close()

    ftest_svm = open(os.path.join('data', 'test.svm.txt'), 'wb')
    dump_svmlight_file(X_test.to_numpy(), [0] * len(X_test), ftest_svm)
    ftest_svm.close()
    """

    RFR_model = RandomForestRegressor(max_depth=10, n_estimators=100)
    RFR_model.fit(X_train, y_train)

    y_pred_RFR = RFR_model.predict(X_test_stdised)
    submission = pd.DataFrame(y_pred_RFR, index=X_test.index, columns=['y'])

    submission.to_csv('out.csv', index=True)

    """
    xgb_model = XGBRegressor()
    xgb_model.fit(X_train, y_train)

    ans = xgb_model.predict(X_test_stdised)
    ftest_csv = open("out.csv", "r")
    fpred_out = open("xgb.csv", "w")

    for idx, line in enumerate(ftest_csv.readlines(), -1):
        if idx == -1:
            fpred_out.write("%s" % line)
        else:
            fpred_out.write('%s,%f\n' % (line.split(",")[0], ans[idx]))
    ftest_csv.close()
    fpred_out.close()
    """
