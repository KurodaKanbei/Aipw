import pandas as pd
import numpy as np
import os
import random
import sklearn
from sklearn.impute import SimpleImputer
import seaborn as sb
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.svm import LinearSVR
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.datasets import dump_svmlight_file
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, SelectFromModel, RFE
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor

if __name__ == '__main__':

    X = pd.read_csv(os.path.join("data", "X_train.csv"),
                    delimiter=',',
                    index_col = 'id')
    y = pd.read_csv(os.path.join("data", "y_train.csv"),
                    delimiter=',',
                    index_col='id')

    idx = []
    for i, feature in enumerate(X.columns):
        values = X[feature].dropna(0)
        mx = values.max()
        mn = values.min()
        if mn > 0 and abs((mx - mn) / mn - .1) < 0.05:
            idx.append(feature)
    X = X.drop(idx, axis=1)
    print(X.shape)
    nan_idx = np.isnan(X)
    # imputer = KNNImputer(n_neighbors=20)
    imputer = SimpleImputer(strategy='median')
    X_imp = imputer.fit_transform(X)

    X_train = X_imp
    y_train = y.to_numpy()

    # lsvr = LinearSVR(C=0.01).fit(X_train, y_train)
    # model = ExtraTreesRegressor(n_estimators=50).fit(X_train, y_train.ravel())
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                        max_depth=3, min_samples_leaf=15, min_samples_split=10)
    selector = RFE(model, step=0.1, n_features_to_select=60, verbose=1)
    # selector = RFECV(estimator, step=0.1,cv=5,min_features_to_select=50,verbose=1)
    selector.fit(X_train, y_train.ravel())
    # selector = SelectKBest(score_func=mutual_info_regression, k=200)
    X_cleaned, y_cleaned = X_train, y_train
    X_train = selector.transform(X_train)

    model = sklearn.ensemble.IsolationForest(contamination="auto")
    outl_pred = model.fit_predict(X_train)
    mask = outl_pred != -1

    X_train, y_train = X_train[mask, :], y_train[mask]

    X_train_backup = X_train.copy()
    X_train = sklearn.preprocessing.StandardScaler().fit(X_train_backup).transform(X_train)

    X_test = pd.read_csv(os.path.join("data", "X_test.csv"),
                         delimiter=',',
                         index_col='id')
    X_test_backup = X_test
    X_test = X_test.drop(idx, axis=1)
    X_test = imputer.fit(X_cleaned).transform(X_test)
    X_test = selector.transform(X_test)
    X_test_stdised = sklearn.preprocessing.StandardScaler().fit(X_train_backup).transform(X_test)
    # X_test_stdised = np.delete(X_test_stdised, cols, axis=1)

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
    print(X_train.shape, y_train.shape)

    # model = RandomForestRegressor(max_depth=10, n_estimators=50)
    model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1,
                                      max_depth=5, max_features=0.3,
                                     min_samples_leaf=15, min_samples_split=10)
    model.fit(X_train, y_train.ravel())
    y_pred_RFR = model.predict(X_test_stdised)
    submission = pd.DataFrame(y_pred_RFR, index=X_test_backup.index, columns=["y"])
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
