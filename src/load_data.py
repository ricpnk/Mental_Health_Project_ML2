import pandas as pd

def load_data():
    """
    Load data from a csv file
    """
    train_data = pd.read_csv("data/train.csv")
    test_data = pd.read_csv("data/test.csv")
    return train_data, test_data