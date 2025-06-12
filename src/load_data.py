import pandas as pd

def load_data():
    """
    Load data from a csv file
    """
    data = pd.read_csv("data/train.csv")
    return data