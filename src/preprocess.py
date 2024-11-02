# src/preprocess.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import sys

def preprocess_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Handle missing values
    data.fillna(0, inplace=True)

    # Assuming 'is_fraud' is the target variable
    X = data.drop('Class', axis=1)
    y = data['Class']

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the preprocessed data
    preprocessed_data_path = os.path.join('data', 'preprocessed_data.pkl')
    pd.to_pickle((X_train_scaled, X_test_scaled, y_train, y_test), preprocessed_data_path)

    print("Preprocessing completed. Data saved to:", preprocessed_data_path)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python preprocess.py data\creditcard.csv")
        sys.exit(1)

    file_path = sys.argv[1]
    preprocess_data(file_path)

