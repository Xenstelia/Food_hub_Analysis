# model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def load_data(file_path):
    """
    Load the FoodHub dataset from a CSV file.
    """
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    """
    Preprocess the FoodHub dataset.
    - Handle missing values if any.
    - Encode categorical variables.
    - Split the data into features (X) and target (y).
    - Split into training and testing sets.
    """
    # Example: Assuming 'Order_Completed' is the target variable
    data = data.dropna()  # Drop rows with missing values
    
    # Encoding categorical variables
    categorical_columns = data.select_dtypes(include=['object']).columns
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
    
    X = data.drop('Order_Completed', axis=1)  # Replace with the actual target column
    y = data['Order_Completed']
    
    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, X.columns

def build_model(X_train, y_train):
    """
    Build and train a Decision Tree model.
    """
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model's performance on the test set.
    """
    y_pred = model.predict(X_test)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))

def visualize_decision_tree(model, feature_names):
    """
    Visualize the trained Decision Tree.
    """
    plt.figure(figsize=(20,10))
    plot_tree(model, feature_names=feature_names, filled=True)
    plt.title("Decision Tree Visualization")
    plt.show()

def feature_importance(model, feature_names):
    """
    Plot the feature importance of the Decision Tree model.
    """
    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10,6))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()