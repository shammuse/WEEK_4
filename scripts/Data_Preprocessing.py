import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from IPython.display import display

class DataProcessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.train_df = None
        self.test_df = None
        self.store_df = None

    def load_data(self):
        self.train_df = pd.read_csv(os.path.join(self.data_dir, 'train.csv'), dtype={'StateHoliday': str}, low_memory=False)
        self.test_df = pd.read_csv(os.path.join(self.data_dir, 'test.csv'), low_memory=False)
        self.store_df = pd.read_csv(os.path.join(self.data_dir, 'store.csv'), low_memory=False)

    def display_data_summary(self):
        print("Train Dataset Head:")
        display(self.train_df.head())

        print("\nTest Dataset Head:")
        display(self.test_df.head())

        print("\nStore Dataset Head:")
        display(self.store_df.head())

        print("\nTrain Dataset Summary:")
        display(self.train_df.describe())

        print("\nTest Dataset Summary:")
        display(self.test_df.describe())

        print("\nStore Dataset Summary:")
        display(self.store_df.describe())
        
    def check_missing_values(self):
        return {
            'train': self.train_df.isnull().sum(),
            'test': self.test_df.isnull().sum(),
            'store': self.store_df.isnull().sum()
        }

    def handle_missing_values(self):
        # Store data missing value handling
        self.store_df['CompetitionDistance'] = self.store_df['CompetitionDistance'].fillna(self.store_df['CompetitionDistance'].median())
        self.store_df['CompetitionOpenSinceMonth'] = self.store_df['CompetitionOpenSinceMonth'].fillna(0)
        self.store_df['CompetitionOpenSinceYear'] = self.store_df['CompetitionOpenSinceYear'].fillna(0)
        self.store_df['Promo2SinceWeek'] = self.store_df['Promo2SinceWeek'].fillna(0)
        self.store_df['Promo2SinceYear'] = self.store_df['Promo2SinceYear'].fillna(0)
        self.store_df['PromoInterval'] = self.store_df['PromoInterval'].fillna('None')

        # Test data missing value handling
        self.test_df['Open'] = self.test_df['Open'].fillna(1)

    def detect_outliers(self):
        train_numeric_cols = self.train_df.select_dtypes(include=[np.number])
        store_numeric_cols = self.store_df.select_dtypes(include=[np.number])
        train_outliers = (np.abs(zscore(train_numeric_cols)) > 3).sum()
        store_outliers = (np.abs(zscore(store_numeric_cols)) > 3).sum()
        return train_outliers, store_outliers

class CustomerBehaviorAnalyzer:
    def __init__(self, train_df):
        self.train_df = train_df

    def visualize_data(self):
        # Box plots for Sales and Customers
        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        sns.boxplot(x=self.train_df['Sales'])
        plt.title('Box Plot of Sales')

        plt.subplot(1, 2, 2)
        sns.boxplot(x=self.train_df['Customers'])
        plt.title('Box Plot of Customers')

        plt.tight_layout()
        plt.show()

        # Histograms for Sales and Customers
        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        plt.hist(self.train_df['Sales'], bins=30, edgecolor='k', color='skyblue')
        plt.title('Histogram of Sales')
        plt.xlabel('Sales')
        plt.ylabel('Frequency')

        plt.subplot(1, 2, 2)
        plt.hist(self.train_df['Customers'], bins=30, edgecolor='k', color='salmon')
        plt.title('Histogram of Customers')
        plt.xlabel('Customers')
        plt.ylabel('Frequency')

        plt.tight_layout()
        plt.show()


