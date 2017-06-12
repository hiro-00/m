import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error, r2_score

# read datasets
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Remove non-onformative columns
cols_to_remove = []
for c in test.columns:
    if len(train[c].unique()) == 1:
        cols_to_remove.append(c)
print('Columns to remove: ' + str(cols_to_remove))
train = train.drop(cols_to_remove, axis=1)
test = test.drop(cols_to_remove, axis=1)

# Add some categorical features
train['X0_0'] = train['X0'].apply(lambda x: x[0])
train['X0_1'] = train['X0'].apply(lambda x: x[1] if len(x) > 1 else 'empty')

test['X0_0'] = test['X0'].apply(lambda x: x[0])
test['X0_1'] = test['X0'].apply(lambda x: x[1] if len(x) > 1 else 'empty')

train['X2_0'] = train['X2'].apply(lambda x: x[0])
train['X2_1'] = train['X2'].apply(lambda x: x[1] if len(x) > 1 else 'empty')

test['X2_0'] = test['X2'].apply(lambda x: x[0])
test['X2_1'] = test['X2'].apply(lambda x: x[1] if len(x) > 1 else 'empty')

# process columns, apply LabelEncoder to categorical features
for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))

##Add decomposed components: PCA / ICA etc.
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD

n_comp = 12

# tSVD
tsvd = TruncatedSVD(n_components=n_comp, random_state=420)
tsvd_results_train = tsvd.fit_transform(train.drop(["y"], axis=1))
tsvd_results_test = tsvd.transform(test)

# PCA
pca = PCA(n_components=n_comp, random_state=420)
pca2_results_train = pca.fit_transform(train.drop(["y"], axis=1))
pca2_results_test = pca.transform(test)

# ICA
ica = FastICA(n_components=n_comp, random_state=420)
ica2_results_train = ica.fit_transform(train.drop(["y"], axis=1))
ica2_results_test = ica.transform(test)

# GRP
grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)
grp_results_train = grp.fit_transform(train.drop(["y"], axis=1))
grp_results_test = grp.transform(test)

# SRP
srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)
srp_results_train = srp.fit_transform(train.drop(["y"], axis=1))
srp_results_test = srp.transform(test)

# Append decomposition components to datasets
for i in range(1, n_comp + 1):
    train['pca_' + str(i)] = pca2_results_train[:, i - 1]
    test['pca_' + str(i)] = pca2_results_test[:, i - 1]

    train['ica_' + str(i)] = ica2_results_train[:, i - 1]
    test['ica_' + str(i)] = ica2_results_test[:, i - 1]

    train['grp_' + str(i)] = grp_results_train[:, i - 1]
    test['grp_' + str(i)] = grp_results_test[:, i - 1]

    train['srp_' + str(i)] = srp_results_train[:, i - 1]
    test['srp_' + str(i)] = srp_results_test[:, i - 1]

    train['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]
    test['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]

y_train = train["y"]
train = train.drop(["y"], axis=1)
y_mean = np.mean(y_train)

### Regressor
from sklearn.ensemble import RandomForestRegressor

# prepare dict of params for xgboost to run with
param = {
    "n_estimators" : 3000,
    "max_features": "sqrt",
    "min_samples_split" : 20,
    "n_jobs": -1
}

def cv():
    cv_num = 3
    nfolds = 5
    score = 0
    for _ in range(cv_num):
        kfold = KFold(n=train.shape[0], n_folds=nfolds, random_state=_, shuffle=True)
        for train_index, test_index in kfold:
            x_train = train.values[train_index]
            label_train = y_train.values[train_index]
            x_test = train.values[test_index]
            label_test =y_train[test_index]

            rf = RandomForestRegressor(**param)
            rf.fit(x_train, label_train)
            y_pred = rf.predict(x_test)

            overfit_test = rf.predict(x_train)
            print("ov:" + str(r2_score(label_train, overfit_test)))
            print(r2_score(y_train.values[test_index], y_pred))
            score += r2_score(label_test, y_pred)
        print("--")
    print(score / (cv_num * nfolds))


def submit():
    rf = RandomForestRegressor(**param)
    rf.fit( train.values, y_train.values)
    y_pred = rf.predict(test.values)

    output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': y_pred})
    output.to_csv('submission_baseLine.csv', index=False)


cv()
submit()

