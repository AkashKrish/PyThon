import pandas as pd
import urllib
import symbol as s
import numpy as np
import pickle
import os

RISK_FREE_RET = 0

def is_non_zero_file(fpath):  
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0

def load_symbol_object():
    '''Load stock historicl prices with default initial values'''
    # Default values initialised
    interval = (0, 0, 5)
    input_file_name='C:\\users\\Akash\\Documents\\Github\\Data Sets\\Yahoo\\nifty50list.csv'
    symbol_list = pd.read_csv(input_file_name)

    # Create a symbol object with the historical data of the stocks in input file
    symbol = s.Symbol(symbol_list=symbol_list, interval=interval)
    return symbol


def save_pickle(object, pickle_path = None):
    '''Save the passed object to the local dataset path'''
    if pickle_path == None:
        pickle_path = 'C:\\users\\Akash\\Documents\\Github\\Data Sets\\Yahoo\\Nifty.pickle'
    
    # Saving the Stock data to a pickle for easier access
    pickle_out = open(pickle_path, 'wb')
    pickle.dump(object, pickle_out)
    pickle_out.close()


def main():
    """Main method for Proceeding"""
    pickle_path = 'C:\\users\\Akash\\Documents\\Github\\Data Sets\\Yahoo\\Nifty.pickle'
    
    print("Lets start")
    #Checking if Pickle file for stock prices is present
    if is_non_zero_file(pickle_path):
        print('Pickle file exists. Grabbing stock data from pickle file')
        pickle_in = open(pickle_path, 'rb')
        symbol = pickle.load(pickle_in)
        pickle_in.close()
    else:
        print('Loading stock data from Yahoo Finance')
        symbol = load_symbol_object()
        save_pickle(symbol, pickle_path)
    tech_symbols = symbol.list.ix[symbol.list['Industry']=='FINANCIAL SERVICES','Symbol']
    tech_symbols = symbol.list[(symbol.list['Industry']=='FINANCIAL SERVICES')]['Symbol']
    
    #print(symbol.data.corr().ix['NSEI',:])
    print(symbol.data[tech_symbols.values].corr())

    symbol.data.to_csv('C:\\users\\Akash\\Documents\\Github\\Data Sets\\Yahoo\\InfyCF.csv')

if __name__ == '__main__':
    import sys
    sys.exit(int(main() or 0))