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
    # clean sleep duration and dietary habits
    valid_sleep_durations = ['Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours']
    default_sleep_duration = '7-8 hours'
    data['Sleep Duration'] = data['Sleep Duration'].apply(lambda x: x if x in valid_sleep_durations else default_sleep_duration)

    valid_dietary_habits = ['Healthy', 'Moderately Healthy', 'Unhealthy']
    default_dietary_habit = 'Moderately Healthy'
    data['Dietary Habits'] = data['Dietary Habits'].apply(lambda x: x if x in valid_dietary_habits else default_dietary_habit)

    # new feature total stress
    for col in ['Academic Pressure', 'Work Pressure', 'Financial Stress']:
        if col in data.columns:
            data[col] = data[col].fillna(0.0)

    data['Total_Stress'] = (
        data['Academic Pressure'] + data['Work Pressure'] + data['Financial Stress']
    )
    data = data.drop(columns=['Academic Pressure', 'Work Pressure', 'Financial Stress'])

    # new feature age stress
    data['Age_Stress'] = data['Total_Stress'] * data['Age']

    # drop unnecessary columns
    drop_columns = ['id', 'Name', 'CGPA']
    drop_columns = [col for col in drop_columns if col in data.columns]
    data = data.drop(columns=drop_columns)

    # create X and y
    feature_columns = data.columns
    feature_columns = feature_columns.drop('Depression')
    target_column = 'Depression'

    X = data[feature_columns]
    y = data[target_column]

    # create categorical, numerical and ordinal columns
    categorical_columns = X.select_dtypes(include=['object', 'category']).columns
    numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns
    ordinal_columns = ['Sleep Duration', 'Dietary Habits']
    categorical_columns = [col for col in categorical_columns if col not in ordinal_columns]

    # Create preprocessing pipeline
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    ordinal_categories = [
        ['Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours'],
        ['Unhealthy', 'Moderately Healthy', 'Healthy']
    ]

    ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(categories=ordinal_categories))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_columns),
            ('ord', ordinal_transformer, ordinal_columns),
            ('num', numerical_transformer, numerical_columns)
        ]
    )

    # Initialize and fit label encoder
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99, stratify=y)

    # preprocess data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Convert DataFrames to tensors
    X_train_tensor = torch.tensor(X_train_processed, dtype=torch.float32)
    Y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32)
    Y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    return train_loader, test_loader