import pandas as pd

def get_mean_volume(symbol):
    """
    return mean volume
    """
    df = pd.read_csv("../datasets/{}.csv".format(symbol))
    return df['Volume'].mean()

def test_run():
    for symbol in ['IBM', 'AAPL', 'GOOG', 'HCP']:
        print("Mean volume")
        print(symbol, get_mean_volume(symbol))

if __name__ == "__main__":
    test_run()

