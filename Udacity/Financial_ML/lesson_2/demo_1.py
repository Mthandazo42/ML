import pandas as pd
import os

def symbol_to_path(symbol, base_dir='../datasets'):
    return os.path.join(base_dir, '{}.csv'.format(str(symbol)))

def get_data(symbols, dates):
    df = pd.DataFrame(index=dates)
    if 'SPY' not in symbols:
        symbols.insert(0, 'SPY')
    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, usecols=['Date','Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
    df = df.dropna()
    return df

def test_run():
    start_date='2019-04-01' #change to the year 2018
    end_date='2019-04-29' #change to the year 2018
    dates=pd.date_range(start_date, end_date)

    #read in more stocks
    symbols = ['GOOG', 'IBM', 'AAPL']
    #get stock data
    df = get_data(symbols, dates)
    #Slice by row range (dates) using DataFrame.ix[] selector
    #print(df.ix['2010-01-01':'2010-01-31']) month of january

    #Slice by column (symbols)
    #print(df['GOOG'] a single label selects a single column
    #print(df[['IBM','AAPL']]) a list of labels selects multiple columns

    #Slice by row and column
    #print(df.ix['2018-03-01':'2018-03-30', ['SPY', 'IBM']])
    print(df)

if __name__ == '__main__':
    test_run()
