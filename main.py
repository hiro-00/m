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

#train = train.drop(883)


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
tsvd = TruncatedSVD(n_components=n_comp, random_state=400)
tsvd_results_train = tsvd.fit_transform(train.drop(["y"], axis=1))
tsvd_results_test = tsvd.transform(test)

# PCA
pca = PCA(n_components=n_comp)# random_state=400)
pca2_results_train = pca.fit_transform(train.drop(["y"], axis=1))
pca2_results_test = pca.transform(test)

# ICA
ica = FastICA(n_components=n_comp)#, random_state=400)
ica2_results_train = ica.fit_transform(train.drop(["y"], axis=1))
ica2_results_test = ica.transform(test)

# GRP
grp = GaussianRandomProjection(n_components=n_comp, eps=0.007, random_state=400)
grp_results_train = grp.fit_transform(train.drop(["y"], axis=1))
grp_results_test = grp.transform(test)

# SRP
srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=400)
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

y_train = train["y"]
train = train.drop(["y"], axis=1)
y_mean = np.mean(y_train)

### Regressor
import xgboost as xgb

# prepare dict of params for xgboost to run with
xgb_params = {
    'n_trees': 50,
    'eta': 0.007,
    'max_depth': 4,
    'subsample': 0.95,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': y_mean,  # base prediction = mean(target)
    'silent': 1
}

# form DMatrices for Xgboost training
#dtest = xgb.DMatrix(test)

#xgboost, cross-validation
def cv():
    num_boost_rounds = 1000
    cv_num = 5
    nfolds = 5
    score = 0
    for _ in range(cv_num):
        kfold = KFold(n=train.shape[0], n_folds = nfolds,  random_state=_, shuffle=True)
        for train_index, test_index in kfold:

            dtrain = xgb.DMatrix(train.values[train_index], label = y_train.values[train_index])
            dtest = xgb.DMatrix(train.values[test_index], label = y_train[test_index])

            model = xgb.train(dict(xgb_params), dtrain, num_boost_round=num_boost_rounds)
            y_pred = model.predict(dtest)

            overfit_test = model.predict(dtrain)
            print("ov:"+str(r2_score(dtrain.get_label(), overfit_test)))
            print(r2_score(y_train.values[test_index], y_pred))
            score += r2_score(dtest.get_label(), y_pred)
        print("--")
    print(score / (cv_num * nfolds) )

def submit():
    num_boost_rounds = 1000

    dtrain = xgb.DMatrix(train.values, label = y_train.values)
    dtest = xgb.DMatrix(test.values)
    model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)


    # print(r2_score(model.predict(dtrain), dtrain.get_label()))

    # make predictions and save results
    y_pred = model.predict(dtest)

    output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': y_pred})
    output.to_csv('submission_baseLine.csv', index=False)

submit()