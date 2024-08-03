import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import logging
from geopy.distance import geodesic
from sklearn.impute import SimpleImputer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data():
    """Load the data files."""
    try:
        calendar_df = pd.read_csv('cleaned_data/cleaned_calendar.csv', low_memory=False)
        listings_df = pd.read_csv('cleaned_data/cleaned_listings.csv', low_memory=False)
        reviews_df = pd.read_csv('cleaned_data/cleaned_reviews.csv', low_memory=False)
        return calendar_df, listings_df, reviews_df
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise
    except pd.errors.EmptyDataError:
        logging.error("One of the CSV files is empty")
        raise

def clean_price(price):
    """Clean price by removing '$' and ',' characters, then convert to float."""
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

def process_calendar_data(df):
    """Process and aggregate calendar data."""
    df['date'] = pd.to_datetime(df['date'])
    df['price'] = df['price'].apply(clean_price)
    df['adjusted_price'] = df['adjusted_price'].apply(clean_price)
    
    # Calculate average price and availability for each listing
    agg_data = df.groupby('listing_id').agg({
        'price': 'mean',
        'available': lambda x: (x == 't').mean(),
        'minimum_nights': 'mean',
        'maximum_nights': 'mean'
    }).rename(columns={
        'price': 'avg_price',
        'available': 'availability_rate',
        'minimum_nights': 'avg_minimum_nights',
        'maximum_nights': 'avg_maximum_nights'
    })
    
    return agg_data

def calculate_proximity_score(lat, lon):
    """Calculate proximity score based on distance to central park."""
    central_park = (40.7829, -73.9654)  # Latitude and longitude of Central Park
    return geodesic((lat, lon), central_park).miles

def process_listings_data(df, calendar_agg):
    """Process listings data and merge with calendar aggregations."""
    # Merge with calendar aggregations
    df = df.merge(calendar_agg, left_on='id', right_index=True, how='left')
    
    # Clean price in listings data
    df['price'] = df['price'].apply(clean_price)
    
    # Create price category
    df['price_category'] = pd.qcut(df['price'], q=3, labels=['low', 'medium', 'high'])
    
    # Encode categorical features
    cat_features = ['neighbourhood_group', 'room_type']
    le = LabelEncoder()
    for feature in cat_features:
        df[f'{feature}_encoded'] = le.fit_transform(df[feature])
    
    # Calculate days since last review
    df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
    df['days_since_last_review'] = (datetime.now() - df['last_review']).dt.days
    
    # Create is_licensed feature
    df['is_licensed'] = df['license'].notna().astype(int)
    
    # Create proximity score
    df['proximity_score'] = df.apply(lambda row: calculate_proximity_score(row['latitude'], row['longitude']), axis=1)
    
    # Create interaction term
    df['neighbourhood_room_type'] = df['neighbourhood'] + '_' + df['room_type']
    
    # Log transform of numeric features
    numeric_features = ['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']
    for feature in numeric_features:
        df[f'log_{feature}'] = np.log1p(df[feature])
    
    # Create time-based features
    df['review_month'] = df['last_review'].dt.month
    df['review_day_of_week'] = df['last_review'].dt.dayofweek
    
    return df

def process_reviews_data(df):
    """Process reviews data to get review frequency."""
    df['date'] = pd.to_datetime(df['date'])
    review_freq = df.groupby('listing_id').agg({
        'id': 'count',
        'date': ['min', 'max']
    })
    review_freq.columns = ['review_count', 'first_review', 'last_review']
    review_freq['review_period_days'] = (review_freq['last_review'] - review_freq['first_review']).dt.days
    review_freq['reviews_per_day'] = review_freq['review_count'] / review_freq['review_period_days'].replace(0, 1)  # Avoid division by zero
    return review_freq[['reviews_per_day']]

def main():
    try:
        logging.info("Starting feature engineering process...")
        calendar_df, listings_df, reviews_df = load_data()
        
        # Process each dataset
        calendar_agg = process_calendar_data(calendar_df)
        review_freq = process_reviews_data(reviews_df)
        
        # Process listings and merge with other data
        final_df = process_listings_data(listings_df, calendar_agg)
        final_df = final_df.merge(review_freq, left_on='id', right_index=True, how='left')
        
        # Separate numeric and categorical columns
        numeric_columns = final_df.select_dtypes(include=[np.number]).columns
        categorical_columns = final_df.select_dtypes(exclude=[np.number]).columns
        
        # Handle missing values
        numeric_imputer = SimpleImputer(strategy='median')
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        
        final_df[numeric_columns] = numeric_imputer.fit_transform(final_df[numeric_columns])
        final_df[categorical_columns] = categorical_imputer.fit_transform(final_df[categorical_columns])
        
        # Save the final dataframe with engineered features
        final_df.to_csv('feature_data/listings_with_engineered_features.csv', index=False)
        logging.info("Feature engineering complete. Updated data saved to 'listings_with_engineered_features.csv'")
        
    except Exception as e:
        logging.error(f"An error occurred during feature engineering: {str(e)}")

if __name__ == "__main__":
    main()