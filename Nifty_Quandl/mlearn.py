'''Module Docstring'''
# pylint: disable=C0103
import numpy as np
from sklearn import model_selection, preprocessing, svm


def create_label(cur_value, fut_value):
    '''Label a value to 1 if Fut Value is greater than current value'''
    if cur_value <= fut_value:
        return 1
    else:
        return 0

def machinize_data(stock_data):
    '''Accept data and output X and y'''
    stock_data['Fut_NSEI'] = stock_data['NSEI'].shift(-1)
    stock_data.dropna(inplace=True)
    stock_data['Label'] = list(map(create_label, stock_data.NSEI, stock_data.Fut_NSEI))

    X = np.array(stock_data.drop(['Fut_NSEI', 'Label'], 1))
    X = preprocessing.scale(X)
    y = np.array(stock_data.Label)

    return X, y


def select_model(X, y, test_size=0.2):
    '''Select train and test data'''
    return model_selection.train_test_split(X, y, test_size=test_size)

def fit_model(stock_data, test_size=0.2):
    '''Fit a model'''
    X, y = machinize_data(stock_data)
    X_train, X_test, y_train, y_test = select_model(X, y, test_size=test_size)

    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    print("Training data score is {}".format(train_score))


    return clf, X_test, y_test

def fit_svg_model(stock_data, test_size=0.2):
    '''TODO: Have to learn'''
    X, y = machinize_data(stock_data)
    X_train, X_test, y_train, y_test = select_model(X, y, test_size=test_size)

    parameters = {'kernel':('linear', 'rbf'), 'C':[1, X_train.shape[1]]}
    svr = svm.SVC()
    clf = model_selection.GridSearchCV(svr, parameters)
    return clf.fit(X_train, y_train), X_test, y_test
