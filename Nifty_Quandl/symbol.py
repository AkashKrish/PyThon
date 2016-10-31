'''Symbol object'''
# pylint: disable=C0103,E1101
import datetime as dt
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
import quandl as qn

qn.ApiConfig.api_key = 'xtxnERQAvJe3f8gro98G'
qn.ApiConfig.api_version = '2015-04-09'


class Symbol(object):
    '''Symbol Object'''
    WINDOW = 20


    '''Object for Portifolio.'''
    def __init__(self, symbol_list, interval, weights=None,  start_value=1):
        '''Construct a portifolio object with daily and annual returns'''
        # If no weights is specified assume equal weightage
        symbols = len(symbol_list)
        if weights is None:
            weights = np.ones(symbols)/symbols

        self._list = symbol_list
        self._interval = interval
        self._weights = weights
        self._data = self.load_symbols(symbol_list, interval)
        self._value = self.get_daily_value(weights=weights, start_value=start_value)
        self._daily_returns = self.get_daily_returns()
        self._annual_returns = self.get_annual_returns()

    @property
    def list(self):
        '''List of tickers'''
        return self._list

    @property
    def interval(self):
        '''Interval of data gathered'''
        return self._interval

    @property
    def weights(self):
        '''Weights of tickers in portifolio'''
        return self._weights

    @property
    def data(self):
        '''Historical ticker price'''
        return self._data

    @property
    def value(self):
        '''Daily value of the symbols'''
        return self._value

    @property
    def daily_returns(self):
        '''Daily returns of tickers'''
        return self._daily_returns

    @property
    def annual_returns(self):
        '''Annual returns of tickers'''
        return self._annual_returns

    @list.setter
    def list(self, symbol_list):
        '''Alter portifolio ticker list'''
        self._list = symbol_list

    @interval.setter
    def interval(self, interval):
        '''Alter interval of data to gather'''
        self._interval = interval

    @weights.setter
    def weights(self, weights):
        '''Alter portifolio weights'''
        self._weights = weights

    @data.setter
    def data(self, symbol_data):
        '''Alter portifolio data'''
        self._data = symbol_data

    @value.setter
    def value(self, new_portifolio):
        '''Alter Portifolio value'''
        self._value = new_portifolio

    def load_symbols(self, symbol_list=None, interval=None):
        '''Loads the data of tickers from yahoo finance into a DataFrame'''
        if symbol_list is None:
            symbol_list = self.list
        if interval is None:
            interval = self.interval

        from_date, to_date = self.get_dates(interval)

        symbol_data = pd.DataFrame(index=pd.date_range(start=from_date, end=to_date))

        for i, symbol in enumerate(symbol_list['Code']):
            data = qn.get(symbol, start_date=from_date, end_date=to_date)
            symbol_data = symbol_data.join(data['Close'], how='left')
            symbol_data.rename(columns={symbol_data.columns[-1]: symbol_list.ix[i, 'Symbol']},
                               inplace=True)

        #
        #return symbol_data.dropna(subset=['NSEI'])
        return symbol_data.dropna()

    def get_dates(self, interval=None):
        '''Generate dates with a specified interval'''
        if interval is None:
            interval == self.interval
        to_date = dt.datetime.today().date()
        from_date = to_date - relativedelta(days=1*interval[0],
                                            months=1*interval[1],
                                            years=1*interval[2])
        return from_date, to_date

    def get_normalised_portifolio(self, data=None):
        '''Normalize price for portifolio based on first day's closing price'''
        if data is None:
            data = self.data
        #BUG: if data[0] = NaN entire column becomes NaN
        normal_df = data/data.apply(lambda x: x[x.first_valid_index()])
        return normal_df

    def get_daily_value(self, data=None, weights=None, start_value=10000):
        '''Calculate the daily value of the portifolio'''
        # If no values are passed assigning defaults
        if data is None:
            data = self.data
        if weights is None:
            weights = self.weights

        # Calculate the daily value
        daily_value = self.get_normalised_portifolio(data) * weights * start_value
        value = pd.DataFrame(daily_value.sum(axis=1), columns=["Value"])
        daily_value = daily_value.join(value, how='left')
        return daily_value

    def get_daily_returns(self, data=None):
        '''Calculate the daily returns of portifolio'''
        if data is None:
            data = self.data
        dret = np.log(data / data.shift(1))
        return dret

    def get_annual_returns(self, data=None):
        '''Calculate annual returns from daily returns'''
        if data is None:
            data = self.data
        daily_returns = self.get_daily_returns(data)
        aret = np.exp(daily_returns.groupby(lambda date: date.year).sum())-1
        return aret

    def get_cret(self, value=None):
        '''Calculate the cumulative returns of portifolio'''
        if value is None:
            value = self.value

        return (value.iloc[-1] / value.iloc[0]) - 1

    def get_rmean(self, data=None):
        '''Get the rolling mean of portifolio'''
        if data is None:
            data = self.data
        return data.rolling(WINDOW).mean()

    def get_rsd(self, data=None):
        '''Get the rolling standard deviation of portifolio'''
        if data is None:
            data = self.data
        return data.rolling(WINDOW).std()

    def get_bsd(self, data=None):
        '''Calculate Bollinger upper and lower bands of portifolio'''
        if data is None:
            data = self.data
        upper_band = data + self.get_rsd(data) * 2
        lower_band = data - self.get_rsd(data) * 2
        return upper_band, lower_band
