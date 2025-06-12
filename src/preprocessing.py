import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from torch.utils.data import DataLoader, TensorDataset


def preprocess_data(data: pd.DataFrame, batch_size):
    """
    Preprocess the data
    """
    # train-test split
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['Depression'])

    # drop unnecessary columns
    drop_columns = ['id', 'Name']
    train_data = train_data.drop(columns=drop_columns)
    test_data = test_data.drop(columns=drop_columns)

    # clean diatary habits
    valid_diatary_habits = ['Healthy', 'Moderately Healthy', 'Unhealthy']
    default_dietary_habit = 'Moderately Healthy'
    train_data['Dietary Habits'] = train_data['Dietary Habits'].apply(lambda x: x if x in valid_diatary_habits else default_dietary_habit)
    test_data['Dietary Habits'] = test_data['Dietary Habits'].map(lambda x: x if x in valid_diatary_habits else default_dietary_habit)

    # clean sleep duration
    valid_sleep_durations = ['Less than 5 hours', '7-8 hours', 'More than 8 hours', '5-6 hours']
    default_sleep_duration = '7-8 hours'
    train_data['Sleep Duration'] = train_data['Sleep Duration'].apply(lambda x: x if x in valid_sleep_durations else default_sleep_duration)
    test_data['Sleep Duration'] = test_data['Sleep Duration'].map(lambda x: x if x in valid_sleep_durations else default_sleep_duration)

    # Define categories for ordinal encoding
    ordinal_categories = [
        ['Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours'],
        ['Unhealthy', 'Moderately Healthy', 'Healthy']
    ]

    # convert categorical columns to numerical
    onehot_list = ['Gender', 'City', 'Working Professional or Student', 'Profession', 'Degree', 'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']
    ordinal_list = ['Sleep Duration', 'Dietary Habits']

    # add preprocessor for one-hot and ordinal encoding
    preprocessor = ColumnTransformer(transformers=[
        ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), onehot_list),
        ('ordinal', OrdinalEncoder(categories=ordinal_categories), ordinal_list)
    ])


    # Split in features and labels
    X_train_data = train_data.drop(columns=['Depression'])
    Y_train_data = train_data['Depression']

    X_test_data = test_data.drop(columns=['Depression'])
    Y_test_data = test_data['Depression']


    # Fit and transform the training data, transform the test data
    X_train_encoded = preprocessor.fit_transform(X_train_data)    
    X_test_encoded = preprocessor.transform(X_test_data)

    # Convert DataFrames to tensors
    X_train_tensor = torch.tensor(X_train_encoded, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train_data.values, dtype=torch.long)

    X_test_tensor = torch.tensor(X_test_encoded, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test_data.values, dtype=torch.long)

    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    return train_loader, test_loader