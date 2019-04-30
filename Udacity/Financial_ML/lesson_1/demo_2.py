import pandas as pd

def get_max_close(symbol):
    """
    Return the maximum closing value for stock indicated by the symbol
    """
    df = pd.read_csv("../datasets/{}.csv".format(symbol))
    return df['Close'].max()

def test_run():
    """
    FUNCTION CALLED BY TEST RUN
    """
    for symbol in ['GOOG', 'IBM', 'AAPL', 'HCP']:
        print("MAX CLOSE")
        print(symbol, get_max_close(symbol))

if __name__ == "__main__":
    test_run()

