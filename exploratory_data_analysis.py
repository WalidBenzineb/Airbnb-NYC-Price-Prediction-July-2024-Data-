# exploratory_data_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data():
    """
    Load cleaned data.
    """
    try:
        calendar_df = pd.read_csv('cleaned_data/cleaned_calendar.csv')
        listings_df = pd.read_csv('cleaned_data/cleaned_listings.csv')
        reviews_df = pd.read_csv('cleaned_data/cleaned_reviews.csv')
        return calendar_df, listings_df, reviews_df
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise
    except pd.errors.EmptyDataError:
        logging.error("One of the CSV files is empty")
        raise
    except pd.errors.ParserError:
        logging.error("Error parsing CSV file")
        raise

def plot_price_distribution(df):
    """
    Plot the distribution of prices with appropriate scale and outlier handling.
    """
    # Remove extreme outliers (e.g., prices above 99.9th percentile)
    price_cutoff = df['price'].quantile(0.999)
    df_filtered = df[df['price'] <= price_cutoff]

    plt.figure(figsize=(12, 6))
    
    # Plot histogram
    sns.histplot(data=df_filtered, x='price', kde=True, color='skyblue')
    
    # Calculate statistics
    mean_price = df_filtered['price'].mean()
    median_price = df_filtered['price'].median()

    # Add vertical lines for mean and median
    plt.axvline(mean_price, color='red', linestyle='dashed', linewidth=2, label=f'Mean: ${mean_price:.2f}')
    plt.axvline(median_price, color='green', linestyle='dashed', linewidth=2, label=f'Median: ${median_price:.2f}')

    plt.title('Distribution of Airbnb Prices in NYC')
    plt.xlabel('Price ($)')
    plt.ylabel('Frequency')
    
    # Set x-axis limit to 99th percentile for better visibility
    plt.xlim(0, df_filtered['price'].quantile(0.99))
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/price_distribution.png')
    plt.close()
    logging.info("Price distribution plot saved")

    # Log some statistics
    logging.info(f"Price statistics:")
    logging.info(f"  Mean: ${mean_price:.2f}")
    logging.info(f"  Median: ${median_price:.2f}")
    logging.info(f"  99th percentile: ${df_filtered['price'].quantile(0.99):.2f}")
    logging.info(f"  Max plotted price: ${price_cutoff:.2f}")

def plot_price_by_room_type(df):
    """
    Plot average price by room type.
    """
    avg_price = df.groupby('room_type')['price'].mean().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    avg_price.plot(kind='bar')
    plt.title('Average Price by Room Type')
    plt.xlabel('Room Type')
    plt.ylabel('Average Price ($)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/price_by_room_type.png')
    plt.close()
    logging.info("Price by room type plot saved")

def plot_correlation_matrix(df):
    """
    Plot correlation matrix of numerical features.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
    plt.title('Correlation Matrix of Numerical Features')
    plt.tight_layout()
    plt.savefig('plots/correlation_matrix.png')
    plt.close()
    logging.info("Correlation matrix plot saved")

def main():
    try:
        calendar_df, listings_df, reviews_df = load_data()

        # Create plots directory
        os.makedirs('plots', exist_ok=True)

        plot_price_distribution(listings_df)
        plot_price_by_room_type(listings_df)
        plot_correlation_matrix(listings_df)

        logging.info("Exploratory Data Analysis complete. Plots saved in 'plots' directory.")
    except Exception as e:
        logging.error(f"An error occurred during the analysis: {str(e)}")

if __name__ == "__main__":
    main()