import pandas as pd

if __name__  == "__main__":
    #take csv from /datasets
    df = pd.read_csv('datasets/data.csv')
    print(df.head())
