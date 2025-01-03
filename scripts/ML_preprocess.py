import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from datetime import timedelta

class RossmannPreprocessor:
    def __init__(self, train_df, store_df):
        self.train_df = train_df
        self.store_df = store_df
        self.scaler = StandardScaler()
        self.le = LabelEncoder()

    def preprocess_datetime(self):
        """Extract date-related features from the 'Date' column."""
        self.train_df['Date'] = pd.to_datetime(self.train_df['Date'])

        # Extracting date features
        self.train_df['Year'] = self.train_df['Date'].dt.year
        self.train_df['Month'] = self.train_df['Date'].dt.month
        self.train_df['Day'] = self.train_df['Date'].dt.day
        self.train_df['WeekOfYear'] = self.train_df['Date'].dt.isocalendar().week
        self.train_df['Weekday'] = self.train_df['Date'].dt.weekday
        self.train_df['IsWeekend'] = self.train_df['Weekday'].apply(lambda x: 1 if x >= 5 else 0)

        # Beginning, mid, and end of the month
        self.train_df['IsMonthStart'] = (self.train_df['Day'] <= 10).astype(int)
        self.train_df['IsMidMonth'] = ((self.train_df['Day'] > 10) & (self.train_df['Day'] <= 20)).astype(int)
        self.train_df['IsMonthEnd'] = (self.train_df['Day'] > 20).astype(int)

        # Days to and after holidays
        holidays = pd.to_datetime(['2015-12-25', '2016-01-01', '2016-04-25'])  # Add more holiday dates
        self.train_df['DaysToHoliday'] = self.train_df['Date'].apply(lambda x: (holidays - x).days.min() if (holidays - x).days.min() >= 0 else 0)
        self.train_df['DaysAfterHoliday'] = self.train_df['Date'].apply(lambda x: (x - holidays).days.min() if (x - holidays).days.min() >= 0 else 0)

    def encode_categorical_features(self):
        """Encode categorical columns."""
        self.train_df['StateHoliday'] = self.le.fit_transform(self.train_df['StateHoliday'])

    def merge_store_data(self):
        """Merge store data with train data."""
        self.train_df = self.train_df.merge(self.store_df, how='left', on='Store')

    def add_sales_related_features(self):
        """Add sales-related features."""
        # Lagged Sales
        self.train_df['Lagged_Sales_1'] = self.train_df['Sales'].shift(1)  # Previous day's sales
        self.train_df['Lagged_Sales_7'] = self.train_df['Sales'].shift(7)  # Sales from the same day last week

        # Rolling Average of Sales
        self.train_df['Rolling_Sales_Mean_7'] = self.train_df['Sales'].rolling(window=7).mean()

    def add_store_specific_features(self):
        """Add store-specific features."""
        # Store Type (assuming it's in store_df)
        self.train_df['StoreType'] = self.store_df['StoreType']  

        # Promo Duration
        self.train_df['PromoDuration'] = (self.train_df['Promo2SinceYear'] - self.train_df['Year']) * 12 + \
                                          (self.train_df['Promo2SinceWeek'] - self.train_df['Month'])

        # Competition Open Days
        self.train_df['Competition_Open_Days'] = (self.train_df['Year'] - self.store_df['CompetitionOpenSinceYear']) * 365 + \
                                                  (self.train_df['Month'] - self.store_df['CompetitionOpenSinceMonth']) * 30

    def feature_scaling(self):
        """Scale numerical features."""
        numeric_features = ['Sales', 'Customers', 'CompetitionDistance', 'Promo2SinceWeek', 'Promo2SinceYear',
                            'Lagged_Sales_1', 'Lagged_Sales_7', 'Rolling_Sales_Mean_7', 'PromoDuration', 'Competition_Open_Days']
        
        # Check for missing columns
        missing_features = [feature for feature in numeric_features if feature not in self.train_df.columns]
        if missing_features:
            print(f"Missing features: {missing_features}")
            return  # Exit the method if any features are missing

        self.train_df[numeric_features] = self.train_df[numeric_features].fillna(self.train_df[numeric_features].median())
        self.train_df[numeric_features] = self.scaler.fit_transform(self.train_df[numeric_features])


    def prepare_for_model(self):
        """Finalize the dataset for model training."""
        feature_columns = ['Store', 'DayOfWeek', 'Customers', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday',
                           'Year', 'Month', 'Day', 'WeekOfYear', 'Weekday', 'IsWeekend', 'IsMonthStart', 
                           'IsMonthEnd', 'IsMidMonth', 'DaysToHoliday', 'DaysAfterHoliday', 
                           'CompetitionDistance', 'Promo2SinceWeek', 'Promo2SinceYear',
                           'Lagged_Sales_1', 'Lagged_Sales_7', 'Rolling_Sales_Mean_7', 'PromoDuration', 'Competition_Open_Days']

        X = self.train_df[feature_columns]
        y = self.train_df['Sales']  # Target variable

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print(f"Data prepared for model training. X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        return X_train, X_test, y_train, y_test
