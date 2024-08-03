# data_cleaning.py

import pandas as pd
import numpy as np
import logging


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_price(price):
    """
    Clean price value by removing '$' and ',' characters if it's a string,
    then convert to float.
    """
    if isinstance(price, str):
        try:
            return float(price.replace('$', '').replace(',', ''))
        except ValueError:
            logging.warning(f"Could not convert price: {price}")
            return np.nan
    elif isinstance(price, (int, float)):
        return float(price)
    else:
        logging.warning(f"Unexpected price type: {type(price)}")
        return np.nan

def clean_calendar(df):
    """
    Clean the calendar dataframe.
    """
    logging.info("Cleaning calendar data...")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['available'] = df['available'].map({'t': True, 'f': False})
    df['price'] = df['price'].apply(clean_price)
    df['adjusted_price'] = df['adjusted_price'].apply(clean_price)
    return df

def clean_listings(df):
    """
    Clean the listings dataframe.
    """
    logging.info("Cleaning listings data...")
    df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
    df['price'] = df['price'].apply(clean_price)
    return df

def clean_reviews(df):
    """
    Clean the reviews dataframe.
    """
    logging.info("Cleaning reviews data...")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df

def main():
    try:
        # Load processed data
        logging.info("Loading processed data...")
        calendar_df = pd.read_csv('data/calendar.csv', low_memory=False)
        listings_df = pd.read_csv('data/listings.csv', low_memory=False)
        reviews_df = pd.read_csv('data/reviews.csv', low_memory=False)

        # Clean data
        clean_calendar_df = clean_calendar(calendar_df)
        clean_listings_df = clean_listings(listings_df)
        clean_reviews_df = clean_reviews(reviews_df)

        # Save cleaned data
        logging.info("Saving cleaned data...")
        clean_calendar_df.to_csv('cleaned_data/cleaned_calendar.csv', index=False)
        clean_listings_df.to_csv('cleaned_data/cleaned_listings.csv', index=False)
        clean_reviews_df.to_csv('cleaned_data/cleaned_reviews.csv', index=False)

        logging.info("Data cleaning complete. Cleaned files saved in 'processed_data' directory.")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise  # Re-raise the exception for debugging purposes

if __name__ == "__main__":
    main()