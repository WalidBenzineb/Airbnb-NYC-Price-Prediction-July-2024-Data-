# feature_exploratory_data_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data():
    """Load the data with engineered features."""
    return pd.read_csv('feature_data/listings_with_engineered_features.csv')

def plot_price_distribution(df):
    """Plot the distribution of prices."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df['price'], kde=True, bins=50)
    plt.title('Distribution of Listing Prices')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.xlim(0, df['price'].quantile(0.99))  # Set x-axis limit to 99th percentile
    plt.savefig('plots/price_distribution.png')
    plt.close()

def plot_feature_correlations(df):
    """Plot correlation heatmap of numerical features."""
    numeric_features = ['price', 'avg_price', 'availability_rate', 'avg_minimum_nights', 
                        'avg_maximum_nights', 'days_since_last_review', 'reviews_per_day']
    corr = df[numeric_features].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Heatmap of Numerical Features')
    plt.savefig('plots/feature_correlations.png')
    plt.close()

def plot_price_vs_availability(df):
    """Plot price vs availability rate."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='availability_rate', y='price', data=df)
    plt.title('Price vs Availability Rate')
    plt.xlabel('Availability Rate')
    plt.ylabel('Price')
    plt.ylim(0, df['price'].quantile(0.99))  # Set y-axis limit to 99th percentile of price
    plt.savefig('plots/price_vs_availability.png')
    plt.close()

def plot_price_by_category(df):
    """Plot price distribution by price category."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='price_category', y='price', data=df)
    plt.title('Price Distribution by Category')
    plt.xlabel('Price Category')
    plt.ylabel('Price')
    plt.ylim(0, df['price'].quantile(0.99))  # Set y-axis limit to 99th percentile of price
    plt.savefig('plots/price_by_category.png')
    plt.close()

def plot_reviews_vs_price(df):
    """Plot reviews per day vs price."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='reviews_per_day', y='price', data=df)
    plt.title('Reviews per Day vs Price')
    plt.xlabel('Reviews per Day')
    plt.ylabel('Price')
    plt.ylim(0, df['price'].quantile(0.99))  # Set y-axis limit to 99th percentile of price
    plt.xlim(0, df['reviews_per_day'].quantile(0.99))  # Set x-axis limit to 99th percentile of reviews_per_day
    plt.savefig('plots/reviews_vs_price.png')
    plt.close()

def summarize_features(df):
    """Print summary statistics of the features."""
    numeric_features = ['price', 'avg_price', 'availability_rate', 'avg_minimum_nights', 
                        'avg_maximum_nights', 'days_since_last_review', 'reviews_per_day']
    summary = df[numeric_features].describe()
    logging.info(f"Summary statistics of numeric features:\n{summary}")

def main():
    try:
        logging.info("Starting feature exploratory data analysis...")
        df = load_data()
        
        summarize_features(df)
        plot_price_distribution(df)
        plot_feature_correlations(df)
        plot_price_vs_availability(df)
        plot_price_by_category(df)
        plot_reviews_vs_price(df)
        
        logging.info("Feature exploratory data analysis complete. Plots saved in 'plots' directory.")
        
    except Exception as e:
        logging.error(f"An error occurred during feature exploratory data analysis: {str(e)}")

if __name__ == "__main__":
    main()