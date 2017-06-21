import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error, r2_score
import pickle

from sklearn.tree import DecisionTreeRegressor as DTC
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from sklearn import tree

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

categorical_columns=[]
# process columns, apply LabelEncoder to categorical features
for c in train.columns:
    if train[c].dtype == 'object':
        categorical_columns.append(c)
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

    train['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]
    test['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]


if True:
    one_host_list = ["X5", "X0_1", "X2_1"]
    for c in one_host_list:
        enc = LabelBinarizer()
        enc.fit(list(train[c].values) + list(test[c].values))
        encoded = pd.DataFrame(enc.transform(list(train[c].values)))
        train = pd.concat([train, encoded], axis=1)
        train.drop(c, axis=1)
        encoded = pd.DataFrame(enc.transform(list(test[c].values)))
        test = pd.concat([test, encoded], axis=1)
        test.drop(c,axis=1)


train = train.drop(['ID'], axis=1)
test = test.drop(['ID'], axis=1)
y_train = train["y"]
y_mean = np.mean(y_train)


### Regressor
import xgboost as xgb

from sklearn.ensemble import RandomForestRegressor
# prepare dict of params for xgboost to run with
xgb_params = {
    'n_trees': 520,
    'eta': 0.0045,
    'max_depth': 4,
    'subsample': 0.98,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': y_mean,  # base prediction = mean(target)
    'silent': 1
}

def outliner_pred():
    outliner = train[train.y > 120]
    # outliner = train[train.y < 150]
    # outliner = outliner[outliner.y > 120]

    y_outliner = outliner['y']
    outliner = outliner.drop(["y"], axis=1)
    outliner.reindex()
    y_outliner.reindex()
    kfold = KFold(n=outliner.shape[0], n_folds=10, random_state=332, shuffle=True)
    score = 0
    mean_score=0
    rf_param = {
        "n_estimators" : 3000,
        #"max_features": "sqrt",
    #    "min_samples_split" : 20,
        "n_jobs": -1
    }


    fig_x=[]
    fig_y=[]
    for train_index, test_index in kfold:
        # dtrain = xgb.DMatrix(outliner.values[train_index], label=y_outliner.values[train_index])
        # dtest  = xgb.DMatrix(outliner.values[test_index], label=y_outliner.values[test_index])
        #
        # model = xgb.train(dict(xgb_params), dtrain, num_boost_round=1000)
        # y_pred = model.predict(dtest)
        # a = mean_squared_error(dtest.get_label(), y_pred)
        # mean_score+=mean_squared_error([np.mean(dtrain.get_label())]*len(test_index), dtest.get_label())
        #
        # fig_x.extend(y_pred)
        # fig_y.extend(dtest.get_label())

        x_train = outliner.values[train_index]
        label_train = y_outliner.values[train_index]
        x_test = outliner.values[test_index]
        label_test = y_outliner.values[test_index]
        rf = RandomForestRegressor(**rf_param)
        rf.fit(x_train, label_train)
        y_pred = rf.predict(x_test)
        fig_x.extend(y_pred)
        fig_y.extend(label_test)
        a = mean_squared_error(label_test, y_pred)
        mean_score+=mean_squared_error([np.mean(label_train)]*len(test_index), label_test)

        score += a
    print(mean_score/5)
    print(score/5)
    print("--")

    plt.scatter(fig_x, fig_y)#,  alpha=0.5)
    plt.xlim([0,200])
    plt.ylim([0,200])
    plt.show()
    exit(0)

train = train.drop(["y"], axis=1)


num_boost_rounds = 1350
def cv():
    cv_num = 1
    nfolds = 10
    score = 0
    fig_x=[]
    fig_y=[]
    for _ in range(cv_num):
        kfold = KFold(n=train.shape[0], n_folds = nfolds,  random_state=_, shuffle=True)
        for train_index, test_index in kfold:

            dtrain = xgb.DMatrix(train.values[train_index], label = y_train.values[train_index])
            dtest = xgb.DMatrix(train.values[test_index], label = y_train[test_index])

            model = xgb.train(dict(xgb_params), dtrain, num_boost_round=num_boost_rounds)
            y_pred = model.predict(dtest)

            #overfit_test = model.predict(dtrain)
            #print("ov:"+str(r2_score(dtrain.get_label(), overfit_test)))
            fig_x.extend(y_pred)
            fig_y.extend(dtest.get_label())
            print(r2_score(y_train.values[test_index], y_pred))
            score += r2_score(dtest.get_label(), y_pred)
        print(score/nfolds)
        print("--")

    if True:
        plt.hist(fig_x,150)
        plt.show()

    if False:
        plt.scatter(fig_x, fig_y)  # ,  alpha=0.5)
        plt.xlim([0, 200])
        plt.ylim([0, 200])
        plt.show()
        print(score / (cv_num * nfolds) )

    train['pred'] = np.array(fig_x)
    if True:
        with open('pred.dump', 'wb') as f:
            pickle.dump(train, f)


def show_tree():
    with open('pred.dump', 'rb') as f:
        train = pickle.load(f)
    thresh = [85, 100, 108]
    row_select=[
        train.pred < thresh[0],
        (train.pred >= thresh[0]) & (train.pred < thresh[1]),
        (train.pred >= thresh[1]) & (train.pred < thresh[2]),
        train.pred >= thresh[2]
    ]

    for i, rows in enumerate(row_select):
        train_rows = train[rows]
        model = DTC(max_depth=5, min_samples_split=int(train_rows.values.shape[0]/7))
        print(train_rows.values.shape)
        model.fit(train_rows.values, y_train[rows].values)
        tree.export_graphviz(model, feature_names=train.columns,out_file="tree/" + "xgb" +str(i)+'tree.dot')
        parse_tree(model, train_rows.values.shape[0], str(i))
        print("")

def parse_tree(estimator, total_node, class_name):
    import collections
    n_nodes = estimator.tree_.node_count

    Node = collections.namedtuple('Node', 'feature thresh rmse size value')
    nodes = []
    print(total_node)
    for n in range(n_nodes):
        if estimator.tree_.n_node_samples[n] > total_node/6:
            nodes.append(Node(feature=estimator.tree_.feature[n],
                              thresh=estimator.tree_.threshold[n],
                              rmse=estimator.tree_.impurity[n],
                              size=estimator.tree_.n_node_samples[n],
                              value=estimator.tree_.value[n]
                              ))
    nodes.sort(key=lambda x:x.rmse/x.size)
    for node in nodes:
        print("class: " + str(class_name))
        print("score: " + str(node.rmse/node.size))
        print("feat : " + str(node.feature))
        print("thresh:" + str(node.thresh))
        print("rmse: " + str(node.rmse))
        print("size: "+str(node.size))
        print("value: "+ str(node.value))
        print("--")

def submit():

    dtrain = xgb.DMatrix(train.values, label = y_train.values)
    dtest = xgb.DMatrix(test.values)
    model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)


    # print(r2_score(model.predict(dtrain), dtrain.get_label()))

    # make predictions and save results
    y_pred = model.predict(dtest)

    output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': y_pred})
    output.to_csv('submission_baseLine.csv', index=False)

#cv()
show_tree()
#submit()
