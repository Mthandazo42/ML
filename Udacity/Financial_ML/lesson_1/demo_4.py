import pandas as pd
import matplotlib.pyplot as plt

def test_run():
    df = pd.read_csv('../datasets/AAPL.csv')
    df[['Close', 'Adj Close', 'High']].plot()
    plt.title("Plot indicating Close, Adj Close and High for Apple Stock")
    plt.show()

if __name__ == "__main__":
    test_run()

