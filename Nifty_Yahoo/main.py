'''Module Docstring'''
# pylint: disable=C0103,E1101
import pickle
import os
import pandas as pd
import numpy as np
import symbol as s
import unsup_mlearn as ml
import cluster as cl


RISK_FREE_RET = 0
def is_non_zero_file(fpath):
    '''Check if the supplied path is  non zero file'''
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0

def load_symbol_object(interval):
    '''Load stock historicl prices with default initial values'''
    # Default values initialised
    input_file_name = '.\\..\\..\\Data Sets\\Yahoo\\nifty50list.csv'
    symbol_list = pd.read_csv(input_file_name)

    # Create a symbol object with the historical data of the stocks in input file
    symbol = s.Symbol(symbol_list=symbol_list, interval=interval)
    return symbol


def save_pickle(obj, pickle_path=None):
    '''Save the passed object to the local dataset path'''
    if pickle_path is None:
        pickle_path = '.\\..\\..\\Github\\Data Sets\\Yahoo\\Nifty.pickle'

    # Saving the Stock data to a pickle for easier access
    pickle_out = open(pickle_path, 'wb')
    pickle.dump(obj, pickle_out)
    pickle_out.close()

def load_all(pickle_path, interval, fresh_load=False):
    '''Load symbol data fro pickle or freshly from Yahoo servers'''
    # Checking if Pickle file for stock prices is present'''
    if is_non_zero_file(pickle_path) and not fresh_load:
        print('Pickle file exists. Grabbing stock data from pickle file')
        pickle_in = open(pickle_path, 'rb')
        symbol = pickle.load(pickle_in)
        pickle_in.close()
    else:
        print('Loading stock data from Yahoo Finance')
        symbol = load_symbol_object(interval)
        save_pickle(symbol, pickle_path)

    return symbol

def should_i_invest(symbol):
    '''Temporary method for '''
    X = symbol.data.copy()
    clf, X_test, y_test = ml.fit_model(X)
    score = clf.score(X_test, y_test)
    # To avoid error the array has to be resized
    future_prices = np.array(X.ix[-1, 0:-2]).reshape(1, -1)
    prediction = clf.predict(future_prices)
    score = score*100 if prediction == 1 else (1-score)*100
    print('You can invest with {}% certainity'.format(round(score, 2)))

def main():
    """Main method for Proceeding"""
    interval = (0, 0, 5)
    pickle_path = '.\\..\\..\\Github\\Data Sets\\Yahoo\\Nifty_10year.pickle'

    print("Lets start")
    symbol = load_all(pickle_path, interval)

    should_i_invest(symbol)

    cl.load_cluster(stock_data=symbol.data.T, n_clusters=5)


if __name__ == '__main__':
    import sys
    sys.exit(int(main() or 0))
