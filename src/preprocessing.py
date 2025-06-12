import pandas as pd
import torch
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from torch.utils.data import DataLoader, TensorDataset


def preprocess_data(data: pd.DataFrame, batch_size):
    """
    Preprocess the data
    """
    # train-test split
    # train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['Depression'])

    # drop unnecessary columns
    drop_columns = ['id', 'Name']
    data = data.drop(columns=drop_columns)

    feature_columns = data.columns[:-1]
    target_column = 'Depression'

    X = data[feature_columns]
    y = data[target_column]

    categorical_columns = X.select_dtypes(include=['object', 'category']).columns
    numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns

    # Create preprocessing pipeline
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_columns),
            ('num', numerical_transformer, numerical_columns)
        ]
    )

    X_processed = preprocessor.fit_transform(X)
    
    # Initialize and fit label encoder
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)
    




    # # clean diatary habits
    # valid_diatary_habits = ['Healthy', 'Moderately Healthy', 'Unhealthy']
    # default_dietary_habit = 'Moderately Healthy'
    # train_data['Dietary Habits'] = train_data['Dietary Habits'].apply(lambda x: x if x in valid_diatary_habits else default_dietary_habit)
    # test_data['Dietary Habits'] = test_data['Dietary Habits'].map(lambda x: x if x in valid_diatary_habits else default_dietary_habit)

    # # clean sleep duration
    # valid_sleep_durations = ['Less than 5 hours', '7-8 hours', 'More than 8 hours', '5-6 hours']
    # default_sleep_duration = '7-8 hours'
    # train_data['Sleep Duration'] = train_data['Sleep Duration'].apply(lambda x: x if x in valid_sleep_durations else default_sleep_duration)
    # test_data['Sleep Duration'] = test_data['Sleep Duration'].map(lambda x: x if x in valid_sleep_durations else default_sleep_duration)

    # # Define categories for ordinal encoding
    # ordinal_categories = [
    #     ['Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours'],
    #     ['Unhealthy', 'Moderately Healthy', 'Healthy']
    # ]

    # # convert categorical columns to numerical
    # onehot_list = ['Gender', 'City', 'Working Professional or Student', 'Profession', 'Degree', 'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']
    # ordinal_list = ['Sleep Duration', 'Dietary Habits']

    # add preprocessor for one-hot and ordinal encoding




    # Convert DataFrames to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    Y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    return train_loader, test_loader