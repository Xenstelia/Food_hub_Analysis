import pandas as pd
import numpy as np

def load_data(filepath):
    """Load the dataset from a CSV file."""
    data = pd.read_csv(filepath)
    return data

def handle_missing_values(data):
    """Handle missing values in the dataset."""
    # Fill missing values for 'Discount' with 0 (assuming no discount if missing)
    if 'Discount' in data.columns:
        data['Discount'].fillna(0, inplace=True)
    
    # Fill missing values for 'Rating' with the median
    if 'Rating' in data.columns:
        data['Rating'].fillna(data['Rating'].median(), inplace=True)
    
    # Fill missing values for categorical columns with the mode
    for column in data.select_dtypes(include=[object]).columns:
        data[column].fillna(data[column].mode()[0], inplace=True)
    
    return data

def detect_and_treat_outliers(data):
    """Detect and treat outliers in the dataset."""
    # Example: Cap outliers for 'Price' and 'Quantity' using the IQR method
    for column in ['Price', 'Quantity']:
        if column in data.columns:
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            data[column] = np.clip(data[column], lower_bound, upper_bound)
    
    return data

def feature_engineering(data):
    """Perform feature engineering on the dataset."""
    # Create a new feature 'Total_Cost' as Price * Quantity
    if 'Price' in data.columns and 'Quantity' in data.columns:
        data['Total_Cost'] = data['Price'] * data['Quantity']
    
    # Create a new feature 'Discounted_Price' as Price - Discount
    if 'Price' in data.columns and 'Discount' in data.columns:
        data['Discounted_Price'] = data['Price'] - data['Discount']
    
    return data

def preprocess_data(filepath):
    """Load and preprocess the data."""
    data = load_data(filepath)
    data = handle_missing_values(data)
    data = detect_and_treat_outliers(data)
    data = feature_engineering(data)
    return data
