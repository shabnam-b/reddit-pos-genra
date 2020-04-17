import glob
import sys
import json
import argparse
from sklearn.metrics import accuracy_score
import numpy as np
import xgboost as xgb
from sklearn import preprocessing

args = argparse.ArgumentParser(description='Program description.')
args.add_argument('-T', '--train', help='Path to train data directory', required=True)
args.add_argument('-Tl', '--train_l', help='Path to a json file containing a list of training labels',required=True)
args.add_argument('-t', '--test', help='Path to test data directory',required=True)
args.add_argument('-tl', '--test_l', help='Path to a json file containing a list of test labels',required=True)
args.add_argument('-Te', '--tre', help='Path to entities for training samples',required=True)
args.add_argument('-te', '--te', help='Path to entities for testing samples',required=True)
args = args.parse_args()


def read_predictions(path):
    preds = []
    labels = []
    files = glob.glob(path + "/*.txt")
    if len(files) == 0:
        print("Please check the directory, no *.txt files were found!")
        sys.exit()
    for f in files:
        tmp = []
        with open(f) as inp:
            for line in inp:
                if line.startswith('\n'):
                    continue
                else:
                    line = line.rstrip().split('\t')
                    tmp.append([line[0]])
                    labels.append(line[0])

        preds.append(tmp)
    return preds, list(set(labels))


def process_features(x_train, x_test, ent_train, ent_test, labels):
    # for tag predictions as features, we use label encoder since it resulted better than one-hot encoding. For
    # entities, we use n-hot encoding (in the KB we used, some tokens were assigned multiple entities).
    l = preprocessing.LabelEncoder()
    l.fit(labels)
    x_train = np.column_stack((k for k in x_train))
    x_test = np.column_stack((k for k in x_test))

    train_encoded = []
    test_encoded = []
    for i, k in enumerate(x_train):
        train_encoded.append(np.append(l.transform(k), ent_train[i]))
    for i, k in enumerate(x_test):
        test_encoded.append((np.append(l.transform(k), ent_test[i])))

    return np.array(train_encoded), np.array(test_encoded)


def train_xgb(x_train, x_test, y_train, y_test):
    le = preprocessing.LabelEncoder()
    le.fit(y_train)
    dtrain = xgb.DMatrix(x_train, label=le.transform(y_train))
    dtest = xgb.DMatrix(x_test)

    # the followings are based on parameter tuning with random search on our dev set.
    param = {
        'max_depth': 7,
        'eta': 0.05,
        'silent': 1,
        'objective': 'multi:softmax',
        'num_class': len(list(set(y_train))),
        "min_child_weight": 1,
        'colsample_bytree': 0.6,
        'gamma': 0,
        'subsample': 0.9}
    print("Training started...")
    bst = xgb.train(param, dtrain, 50)
    predictions = bst.predict(dtest)
    predictions = [int(x) for x in predictions]
    predictions = le.inverse_transform(predictions)
    print('Model acc:%1.6f' % accuracy_score(y_test, predictions))
    return predictions


if __name__ == "__main__":
    with open(args.train_l, 'r') as l:
        y_train = json.load(l)
    with open(args.test_l, 'r') as l:
        y_test = json.load(l)

    with open(args.tre, 'r') as e:
        entity_train = json.load(e)
    with open(args.te, 'r') as e:
        entity_test = json.load(e)
    train, lb = read_predictions(args.train)

    test, _ = read_predictions(args.test)
    train, test = process_features(train, test, entity_train, entity_test, lb+y_train)
    predictions = train_xgb(train,test,y_train,y_test)
