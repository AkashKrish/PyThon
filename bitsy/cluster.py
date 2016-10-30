'''Clustering the survivors of titanic'''
#pylint: disable=E1101,C0103
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="white", palette="muted", color_codes=True)


def cleanse_data(titanic_data, dropna=False):
    '''Cleanse data with binning numerical data and digitize categorical data'''
    if dropna:
        titanic_data.dropna(inplace=True)

    #Grab the type of the cabin
    titanic_data['Cabin'] = titanic_data.Cabin.dropna().apply(lambda x: str(x)[0])

    #Binning numerical data into Bins
    bins = np.linspace(40, 200, 10)
    titanic_data.Fare = np.digitize(titanic_data.Fare, bins)
    bins = np.linspace(0, 60, 6)
    titanic_data.Age = np.digitize(titanic_data.Age, bins)
    bins = np.linspace(0, 6, 3)
    titanic_data.Parch = np.digitize(titanic_data.Parch, bins)

    #Digitize data by converting caegorical data into numerical data
    for column in titanic_data.columns:
        if titanic_data[column].dtype is not int:
            new = pd.Categorical(titanic_data[column])
            titanic_data[column] = (new.codes)

    #titanic_data[column] = titanic_data[column].apply(lambda x: -99999 if x == -1 else x)

    return titanic_data


def load_titanic_data(data_path):
    '''Load the data present in DataSet'''
    titanic_data = pd.read_csv(data_path)

    # Error handler for test data wich doest done have survived field
    if 'Survived' in titanic_data.columns:
        titanic_data.dropna(subset=['Survived'], inplace=True)

    titanic_data.drop(['PassengerId', 'Name', 'Ticket'], 1, inplace=True)
    titanic_data = cleanse_data(titanic_data)
    return titanic_data

def load_cluster(titanic_data):
    '''Load and fit cluster from the data'''
    X = np.array(titanic_data.drop(['Survived'], 1))
    y = np.array(titanic_data.Survived)

    cluster = KMeans(n_clusters=2)
    cluster.fit(X)
    return cluster, X, y

def score_cluster(cluster, X, y):
    '''testing the cluster by usng the same data'''
    correct = 0
    for i, _ in enumerate(X):
        predict_me = np.array(X[i])
        predict_me = predict_me.reshape(-1, len(predict_me))
        prediction = cluster.predict(predict_me)
        if prediction[0] == y[i]:
            correct += 1

    accuracy = correct/len(X)
    print(accuracy, 1-accuracy)

def predict_survival(cluster, titanic_test_data):
    '''Predict survival and join it with test data dataframe'''
    survival = cluster.predict(titanic_test_data)
    survival = pd.DataFrame(survival, columns=['Survived'])
    titanic_test_data = titanic_test_data.join(survival)
    print(titanic_test_data.Survived.describe())

def main():
    '''Start'''
    train_data_path = 'C:\\users\\Akash\\Documents\\Github\\Data Sets\\Kaggle\\titanic_train.csv'
    test_data_path = 'C:\\users\\Akash\\Documents\\Github\\Data Sets\\Kaggle\\titanic_test.csv'

    titanic_data = load_titanic_data(train_data_path)
    cluster, X, y = load_cluster(titanic_data)

    titanic_test_data = load_titanic_data(test_data_path)
    score_cluster(cluster, X, y)
    predict_survival(cluster, titanic_test_data)

    labels = cluster.labels_
    plt.scatter(X[:, 0], X[:,1], c=labels)

    #plt.show()


if __name__ == '__main__':
    import sys
    print(__doc__)
    sys.exit(int(main() or 0))
