import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import logging

# Ensure the visual aesthetics of plots
sns.set(style="whitegrid")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
class SalesAnalyzer:
    def __init__(self, train_df, test_df, store_df):
        self.train_df = train_df
        self.test_df = test_df
        self.store_df = store_df
        logging.info("SalesAnalyzer initialized with training and test data.")

    def plot_promo_distribution(self):
        logging.info("Plotting promo distribution.")
        plt.figure(figsize=(10, 5))

        # Training set
        plt.subplot(1, 2, 1)
        sns.countplot(data=self.train_df, x='Promo', hue='Promo', palette='pastel', legend=False)
        plt.title('Promo Distribution in Training Set')

        # Test set
        plt.subplot(1, 2, 2)
        sns.countplot(data=self.test_df, x='Promo', hue='Promo', palette='pastel', legend=False)
        plt.title('Promo Distribution in Test Set')

        plt.tight_layout()
        plt.show()


    def analyze_sales_holidays(self):
        logging.info("Analyzing sales during holidays.")
        sales_holiday = self.train_df.groupby(['StateHoliday', 'SchoolHoliday'])['Sales'].mean().reset_index()
        plt.figure(figsize=(10, 5))
        sns.barplot(data=sales_holiday, x='StateHoliday', y='Sales', hue='SchoolHoliday', palette='muted')
        plt.title('Average Sales during Holidays')
        plt.show()

    def analyze_sales_by_month(self):
        logging.info("Analyzing sales by month.")
        self.train_df['Month'] = pd.to_datetime(self.train_df['Date']).dt.month
        sales_by_month = self.train_df.groupby('Month')['Sales'].mean().reset_index()
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=sales_by_month, x='Month', y='Sales', marker='o', color='blue')
        plt.title('Average Sales by Month (Seasonal Behavior)')
        plt.show()

    def correlation_analysis(self):
        logging.info("Performing correlation analysis.")
        correlation = self.train_df[['Sales', 'Customers']].corr()
        print("Correlation between Sales and Number of Customers:")
        print(correlation)
        return correlation

    def ols_regression(self):
        logging.info("Performing OLS regression.")
        X = self.train_df[['Customers']]
        y = self.train_df['Sales']
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        print(model.summary())

    def compare_promo_sales(self):
        logging.info("Comparing sales with and without promotions.")
        promo_sales = self.train_df.groupby('Promo')['Sales'].mean().reset_index()
        plt.figure(figsize=(8, 5))
        plt.bar(promo_sales['Promo'], promo_sales['Sales'], color=['skyblue', 'salmon'])
        plt.xticks([0, 1], ['No Promo', 'Promo'])
        plt.title('Sales with vs without Promotions')
        plt.xlabel('Promo')
        plt.ylabel('Average Sales')
        plt.show()

    def analyze_customer_categories(self):
        logging.info("Analyzing sales by customer categories.")
        self.train_df['CustomerCategory'] = pd.qcut(self.train_df['Customers'], q=3, labels=['Small', 'Medium', 'Large'])
        promo_customer_sales = self.train_df.groupby(['Promo', 'CustomerCategory'], observed=False)['Sales'].mean().unstack()
        promo_customer_sales.plot(kind='bar', stacked=True, figsize=(10, 6), color=['lightgreen', 'lightblue', 'salmon'])
        plt.title('Promo Effectiveness on Sales (Small, Medium, Large Customers)')
        plt.xlabel('Promo')
        plt.ylabel('Average Sales')
        plt.legend(title='Customer Category')
        plt.show()

    def analyze_assortment_sales(self):
        logging.info("Analyzing sales by assortment type.")
        merged_df = pd.merge(self.train_df, self.store_df[['Store', 'Assortment']], on='Store', how='left')
        assortment_sales = merged_df.groupby('Assortment')['Sales'].mean().reset_index()
        plt.figure(figsize=(8, 5))
        sns.barplot(data=assortment_sales, x='Assortment', y='Sales', hue='Assortment', palette='viridis', legend=False)
        plt.title('Average Sales by Assortment Type')
        plt.ylabel('Average Sales')
        plt.xlabel('Assortment Type')
        plt.show()
        
    def analyze_sales_by_competition_distance(self):
        logging.info("Analyzing sales by customer categories based on competition distance.")
        
        # Merge train_df with store_df on the Store column
        merged_df = self.train_df.merge(self.store_df[['Store', 'CompetitionDistance']], on='Store', how='left')
        
        # Grouping sales by competition distance
        competition_sales = merged_df.groupby('CompetitionDistance')['Sales'].mean().reset_index()
        
        # Plotting the average sales by competition distance
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=competition_sales, x='CompetitionDistance', y='Sales', color='Red')
        plt.title('Average Sales by Competition Distance')
        plt.xlabel('Competition Distance')
        plt.ylabel('Average Sales')
        plt.show()

    def analyze_city_center_stores(self):
        logging.info("Analyzing average sales by competition distance for city center stores.")

        # Merge train_df with store_df on 'Store'
        merged_df = self.train_df.merge(self.store_df, on='Store', how='left')

        # Filter for city center stores (assuming 'a' indicates city centers)
        city_center_stores = merged_df[merged_df['StoreType'] == 'a']

        # Grouping sales by competition distance for city center stores
        competition_sales_city_center = city_center_stores.groupby('CompetitionDistance')['Sales'].mean().reset_index()

        # Plotting the average sales by competition distance for city center stores
        plt.figure(figsize=(10, 5))
        sns.barplot(data=competition_sales_city_center, x='CompetitionDistance', y='Sales', hue='CompetitionDistance', palette='coolwarm', legend=False)
        plt.title('Average Sales vs Competition Distance for City Center Stores')
        plt.xlabel('Competition Distance')
        plt.ylabel('Average Sales')
        plt.show()

    def monthly_sales_trend(self):
        logging.info("Analyzing monthly sales trend.")
        self.train_df['Date'] = pd.to_datetime(self.train_df['Date'])
        monthly_sales = self.train_df.resample('ME', on='Date')['Sales'].mean().reset_index()
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=monthly_sales, x='Date', y='Sales', color='orange')
        plt.title('Average Monthly Sales Trend')
        plt.xlabel('Month')
        plt.ylabel('Average Sales')
        plt.xticks(rotation=45)
        plt.show()

    def plot_sales_vs_customers(self):
        logging.info("Plotting sales vs. number of customers.")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.train_df, x='Customers', y='Sales', color='red')
        plt.title('Scatter Plot of Sales vs. Number of Customers')
        plt.xlabel('Number of Customers')
        plt.ylabel('Sales')
        plt.show()