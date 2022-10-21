# ref: https://doi.org/10.1002/minf.201300161

import numpy as np
import pandas as pd
from sklearn.base import clone


def mdi(model, X_train, y_train, X_test, y_test=None):
    """
    Parameters
    ----------
    model : scikit-learn estimator object
        A trained model.

    X_train : pandas Dataframe (n_samples, n_features)
        Feature matrix of training set, where `n_samples` is the number of samples and
        `n_features` is the number of predictors.

    y_train : pandas Series (n_samples)
        Target vector of training set, where `n_samples` is the number of samples.

    X_test : pandas Dataframe (n_samples, n_features)
        Feature matrix of test set, where `n_samples` is the number of samples and
        `n_features` is the number of predictors.

    y_test : pandas Series (n_samples), default=None
        Target vector of test set, where `n_samples` is the number of samples.
        If y_test is None returns MDI.
        If y_test is not None returns (MDI, PE) tuple.

    Returns
    -------
    MDI : pandas Series (n_samples in X_test)
        The MDI value of samples in test set.

    PE : pandas Series (n_samples in X_test)
        Prediction error of samples in test set.

    """
    mdi = pd.Series(dtype="float64")
    pe = pd.Series(dtype="float64")
    m2 = clone(model)

    X = pd.concat([X_train, X_test])
    coef = pd.DataFrame(np.corrcoef(X), index=X.index, columns=X.index)
    coef_tr_te = coef.loc[X_train.index, X_test.index]
    mdi = pd.Series(dtype="float64")

    for j in X_test.index:
        r = coef_tr_te[j].idxmax()

        # replace xr by xj
        X_train_new = X_train.copy()
        X_train_new.loc[r] = X_test.loc[j]

        y_train_new = y_train.copy()
        y_calj_test = model.predict(X_train_new.loc[r].to_frame().T).reshape(-1)[0]
        y_train_new.loc[r] = y_calj_test

        # refit model
        m2.fit(X_train_new, y_train_new)
        y_calj_tr_new = m2.predict(X_train_new).reshape(-1)

        # calc MDI and PE
        mdi.loc[j] = np.sum(np.abs(y_cal_tr - y_calj_tr_new))
        if y_test is not None:
            pe.loc[j] = np.abs(y_test[j] - y_calj_test)

    if y_test is not None:
        return (mdi, pe)
    else:
        return mdi
