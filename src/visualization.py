import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_order_distribution(data):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Order_Status', data=data)
    plt.title('Distribution of Order Status')
    plt.xlabel('Order Status')
    plt.ylabel('Count')
    plt.show()

def plot_income_vs_order_value(data):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Income', y='Order_Value', hue='Order_Status', data=data)
    plt.title('Income vs Order Value')
    plt.xlabel('Income')
    plt.ylabel('Order Value')
    plt.legend(title='Order Status')
    plt.show()

def plot_correlation_heatmap(data):
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.show()

# Example usage:
# data = pd.read_csv('foodhub_order.csv')
# plot_order_distribution(data)
# plot_income_vs_order_value(data)
# plot_correlation_heatmap(data)