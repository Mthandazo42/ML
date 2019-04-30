#organize our imports

import pandas as pd

def test_run(filename):
    df = pd.read_csv(filename)
    print('HEAD: ', df.head())
    print('Tail: ', df.tail())
    print('Range example: ', df[10:21])

if __name__ == "__main__":
    test_run("../datasets/HCP.csv")
