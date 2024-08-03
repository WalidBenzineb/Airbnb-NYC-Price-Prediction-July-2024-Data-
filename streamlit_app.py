import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the model and data
@st.cache_resource
def load_model_and_data():
    model = joblib.load('models/best_price_prediction_model.joblib')
    data = pd.read_csv('feature_data/listings_with_engineered_features.csv')
    return model, data

model, data = load_model_and_data()

# Cache data processing functions
@st.cache_data
def process_data_for_viz(data):
    data['price_category'] = pd.qcut(data['price'], q=4, labels=['Budget', 'Economy', 'Standard', 'Luxury'])
    data['review_bins'] = pd.cut(data['number_of_reviews'], bins=[0, 10, 50, 100, np.inf], labels=['0-10', '11-50', '51-100', '100+'])
    return data

# Streamlit app
st.title('Airbnb Price Prediction and Analysis')

# Sidebar for navigation
page = st.sidebar.radio("Navigate", ["Price Prediction", "Data Exploration"])

if page == "Price Prediction":
    st.header('Price Prediction')

    # Create input fields for user to enter listing details
    st.subheader('Enter Listing Details')

    
    neighbourhood = st.selectbox('Neighbourhood', data['neighbourhood'].unique())
    room_type = st.selectbox('Room Type', data['room_type'].unique())
    minimum_nights = st.number_input('Minimum Nights', min_value=1, max_value=365, value=1)
    number_of_reviews = st.number_input('Number of Reviews', min_value=0, value=0)
    reviews_per_month = st.number_input('Reviews per Month', min_value=0.0, value=0.0)
    calculated_host_listings_count = st.number_input('Host Listings Count', min_value=1, value=1)
    availability_365 = st.slider('Availability (days per year)', min_value=0, max_value=365, value=365)
    
    # Create a dataframe from user input
    user_input = pd.DataFrame({
        'neighbourhood': [neighbourhood],
        'room_type': [room_type],
        'minimum_nights': [minimum_nights],
        'number_of_reviews': [number_of_reviews],
        'reviews_per_month': [reviews_per_month],
        'calculated_host_listings_count': [calculated_host_listings_count],
        'availability_365': [availability_365]
    })

    # Calculate avg_price based on neighbourhood and room_type
    avg_price = data[(data['neighbourhood'] == neighbourhood) & (data['room_type'] == room_type)]['price'].mean()
    user_input['avg_price'] = avg_price

    # Add engineered features
    user_input['availability_rate'] = user_input['availability_365'] / 365
    user_input['price_per_night'] = user_input['avg_price'] / user_input['minimum_nights'].clip(lower=1)
    user_input['is_superhost'] = (user_input['calculated_host_listings_count'] > 1).astype(int)
    user_input['high_availability'] = (user_input['availability_365'] > 180).astype(int)

    # Ensure all necessary columns are present
    for col in model.feature_names_in_:
        if col not in user_input.columns:
            if col in data.columns:
                if data[col].dtype == 'object':
                    user_input[col] = data[col].mode().iloc[0]
                else:
                    user_input[col] = data[col].median()
            else:
                user_input[col] = 0  # Default value if column is not in the dataset

    # Encode categorical variables
    le = LabelEncoder()
    for col in user_input.select_dtypes(include=['object']):
        le.fit(data[col])
        user_input[col] = le.transform(user_input[col])

    # Make prediction
    if st.button('Predict Price'):
        try:
            user_input = user_input[model.feature_names_in_]
            prediction = np.expm1(model.predict(user_input))
            st.success(f'The predicted price is ${prediction[0]:.2f} per night')
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")

elif page == "Data Exploration":
    st.header('Data Exploration')
    
    # Process data for visualization only when Data Exploration is selected
    data = process_data_for_viz(data)

    # User controls for plot visibility
    
    show_availability_dist = st.checkbox("Show Availability Distribution", value=True)
    show_review_price = st.checkbox("Show Price vs Reviews", value=True)
    show_room_type = st.checkbox("Show Price by Room Type", value=True)
    show_neighborhood = st.checkbox("Show Top Neighborhoods", value=True)
    show_correlation = st.checkbox("Show Correlation Heatmap", value=True)
    show_min_nights = st.checkbox("Show Price vs Minimum Nights", value=True)


    if show_availability_dist:
        st.subheader('Availability Distribution by Price Category')
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(x='price_category', y='availability_365', data=data, ax=ax)
        plt.title('Availability Distribution by Price Category')
        plt.xlabel('Price Category')
        plt.ylabel('Availability (days per year)')
        st.pyplot(fig)

    if show_review_price:
        st.subheader('Price Distribution by Number of Reviews')
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(x='review_bins', y='price', data=data, ax=ax)
        plt.title('Price Distribution by Number of Reviews')
        plt.xlabel('Number of Reviews')
        plt.ylabel('Price')
        plt.ylim(0, data['price'].quantile(0.95))
        st.pyplot(fig)

    if show_room_type:
        st.subheader('Price Distribution by Room Type')
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(x='room_type', y='price', data=data, ax=ax)
        plt.title('Price Distribution by Room Type')
        plt.xlabel('Room Type')
        plt.ylabel('Price')
        plt.ylim(0, data['price'].quantile(0.95))
        st.pyplot(fig)

    if show_neighborhood:
        st.subheader('Average Price by Neighborhood (Top 15)')
        top_neighborhoods = data.groupby('neighbourhood')['price'].mean().nlargest(15).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(12, 8))
        top_neighborhoods.plot(kind='barh', ax=ax)
        plt.title('Top 15 Neighborhoods by Average Price')
        plt.xlabel('Average Price')
        plt.ylabel('Neighborhood')
        st.pyplot(fig)

    if show_correlation:
        st.subheader('Correlation Heatmap')
        numeric_cols = ['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 
                        'calculated_host_listings_count', 'availability_365']
        corr = data[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        plt.title('Correlation Heatmap of Numeric Features')
        st.pyplot(fig)

    if show_min_nights:
        st.subheader('Price vs. Minimum Nights')
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.regplot(x='minimum_nights', y='price', data=data, scatter=True, scatter_kws={'alpha':0.3}, ax=ax)
        plt.title('Price vs. Minimum Nights')
        plt.xlabel('Minimum Nights')
        plt.ylabel('Price')
        plt.xlim(0, data['minimum_nights'].quantile(0.99))
        plt.ylim(0, data['price'].quantile(0.95))
        st.pyplot(fig)

# Run this app with `streamlit run streamlit_app.py`