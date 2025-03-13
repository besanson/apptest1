import streamlit as st
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from faker import Faker
import random

st.set_page_config(layout="wide")
st.title("📍 AI-Powered Store Location Optimization Demo")

# Initialize Faker
fake = Faker()

# Generate synthetic data (this would typically be loaded externally)
@st.cache_data
def generate_synthetic_data(num_locations=100):
    data = []
    for i in range(num_locations):
        data.append({
            'location_id': f"LOC_{i+1}",
            'latitude': np.random.uniform(25.75, 25.85),
            'longitude': np.random.uniform(-80.25, -80.15),
            'avg_income': np.random.normal(50000, 15000),
            'foot_traffic': np.random.randint(500, 5000),
            'competitor_density': np.random.randint(0, 15),
            'rent_price': np.random.uniform(20, 80),
            'sentiment_score': round(np.random.uniform(0.4, 0.9), 2),
            'demographic_match': round(np.random.uniform(5, 10), 1)
        })
    df = pd.DataFrame(data)

    # GPT-like synthetic reviews
    reviews = [
        "Convenient location, perfect for quick visits during work breaks.",
        "Crowded but vibrant area, coffee quality makes it worth it.",
        "Great coffee but parking can be difficult.",
        "Ideal location near offices, slightly expensive but worth it.",
        "Average experience due to heavy competition nearby."
    ]
    df['synthetic_review'] = [random.choice(reviews) for _ in range(num_locations)]

    # Monthly sales calculation
    df['monthly_sales'] = (
        df['foot_traffic'] * np.random.uniform(5,15) +
        df['demographic_match'] * 1000 -
        df['competitor_density'] * 800 +
        df['sentiment_score'] * 2000 -
        df['rent_price'] * 500 +
        np.random.normal(0, 1000, num_locations)
    ).astype(int)

    return df

data = generate_synthetic_data()

# Train predictive model
@st.cache_resource
def train_model(df):
    features = ['avg_income', 'foot_traffic', 'competitor_density', 
                'rent_price', 'sentiment_score', 'demographic_match']
    X = df[features]
    y = df['monthly_sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(n_estimators=200, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model(data)

# Sidebar inputs for demo interaction
st.sidebar.header("🔍 Search Criteria")
center_lat = st.sidebar.number_input("Center Latitude", value=25.80, format="%.5f")
center_lon = st.sidebar.number_input("Center Longitude", value=-80.20, format="%.5f")
radius_km = st.sidebar.slider("Search Radius (km)", min_value=0.5, max_value=5.0, value=2.0, step=0.1)

# Function to calculate distance
from geopy.distance import geodesic

def within_radius(row, center, radius):
    loc = (row['latitude'], row['longitude'])
    distance = geodesic(center, loc).km
    return distance <= radius

center_point = (center_lat, center_lon)
filtered_data = data[data.apply(within_radius, axis=1, center=center_point, radius=radius_km)]

# Predict sales for filtered locations
filtered_data['predicted_sales'] = model.predict(filtered_data[['avg_income', 'foot_traffic', 'competitor_density', 
                                                                'rent_price', 'sentiment_score', 'demographic_match']]).astype(int)

# Rank recommendations
recommendations = filtered_data.sort_values(by='predicted_sales', ascending=False).head(5)

# Display interactive map
st.subheader("🗺️ Location Map")
st.map(filtered_data[['latitude', 'longitude']])

# Display top recommendations
st.subheader("🏅 Top Recommended Locations")
st.dataframe(recommendations[['location_id', 'predicted_sales', 'foot_traffic', 
                              'competitor_density', 'demographic_match', 'synthetic_review']], 
             use_container_width=True)

# Display detailed GPT-like explanations
st.subheader("💬 GPT-generated Recommendation Explanations")
for idx, row in recommendations.iterrows():
    with st.expander(f"📌 {row['location_id']} - Sales Prediction: ${row['predicted_sales']}"):
        explanation = f"""
        **Why this location?**

        - **High Predicted Monthly Sales:** ${row['predicted_sales']}
        - **Foot Traffic:** {row['foot_traffic']} visitors/week.
        - **Competitor Density:** {row['competitor_density']} nearby.
        - **Demographic Match:** {row['demographic_match']}/10.
        
        **Consumer Feedback:**
        _"{row['synthetic_review']}"_

        This combination strongly suggests high market potential and profitability.
        """
        st.markdown(explanation)

