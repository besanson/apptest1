import streamlit as st
import pandas as pd
import numpy as np
from faker import Faker
import xgboost as xgb
from sklearn.model_selection import train_test_split
from geopy.distance import geodesic

# App Configuration
st.set_page_config(layout="wide", page_title="Poesia AI Location Advisor")
st.title("🍽️ Poesia AI-Powered Location Advisor")
st.markdown("##### Optimizing New Store Locations in Barcelona")

fake = Faker()

# Generate synthetic data for Barcelona
@st.cache_data
def generate_data(num_locations=100):
    data = []
    for i in range(num_locations):
        data.append({
            'location_id': f"LOC_{i+1}",
            'latitude': np.random.uniform(41.36, 41.42),
            'longitude': np.random.uniform(2.14, 2.19),
            'avg_income': np.random.normal(40000, 10000),
            'foot_traffic': np.random.randint(500, 5000),
            'competitor_density': np.random.randint(0, 10),
            'rent_price': np.random.uniform(30, 100),
            'demographic_match': round(np.random.uniform(6,10),1),
            'sentiment_score': round(np.random.uniform(0.5,0.9),2)
        })
    return pd.DataFrame(data)


data = generate_data()

# Model Training
@st.cache_resource(show_spinner="Training AI Model...")
def train_model(df):
    X = df[['demographic_match', 'foot_traffic', 'competitor_density', 'rent_price', 'sentiment_score']]
    y = (df['foot_traffic'] * 10 + df['demographic_match']*1000 -
         df['competitor_density']*1200 + df['sentiment_score']*1000 - df['rent_price']*300).astype(int)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(n_estimators=30, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_ai_model(data)

# Sidebar for CEO interaction
with st.sidebar:
    st.image("https://static.wixstatic.com/media/0321ec_ee3b53f39a3f44c9a6fdbd53c903afc5~mv2.png", width=150)
    st.markdown("### 🔎 Store Preferences")
    radius = st.slider("Search Radius (km)", 0.5, 5.0, 2.0, step=0.1)
    store_type = st.selectbox("Store Type", ["Flagship", "Normal", "Corner"])
    goal = st.radio("Primary Objective", ["Revenue 📈", "Brand Awareness 💡", "Foot Traffic 🚶"])

# Poesia location
poesia_coords = (41.400893, 2.152517)  # Gran de Gràcia, 164, Barcelona

# Calculate distances and filter locations
data['distance_km'] = data.apply(lambda row: geodesic(poesia_coords, (row['latitude'], row['longitude'])).km, axis=1)
filtered = data[data['distance_km'] <= radius].copy()

# Predict sales
features = ['demographic_match', 'foot_traffic', 'competitor_density', 'rent_price', 'sentiment_score']
filtered_X = data[features := features]
model_predictions = model.predict(filtered_X := filtered_X := data[features]).astype(int)
filtered_data = data.assign(predicted_sales=model_predictions)

# User selects primary target
primary_target = st.sidebar.selectbox("Primary Target 🎯", ["Revenue", "Brand Awareness", "Foot Traffic"])

# Score calculation based on user choice
weights = {
    'Revenue': {'sales':0.6, 'demographic_match':0.2, 'foot_traffic':0.2},
    'Brand Awareness': {'sales':0.2,'foot_traffic':0.3,'sentiment_score':0.5},
    'Foot Traffic': {'sales':0.2, 'foot_traffic':0.6, 'brand_awareness':0.2}
}

w = weights.get(primary_target, {'sales':0.5,'foot_traffic':0.3,'brand_awareness':0.2})

filtered_data['score'] = (
    filtered_data['predicted_sales'] * w.get('sales',0.5) +
    filtered_data['foot_traffic'] * w.get('foot_traffic',0.3) * 2 +
    filtered_data['demographic_match'] * 500
)

top_recommendations = filtered_data.sort_values(by='score', ascending=False).head(5)

# UI - Map visualization
st.subheader(f"📍 Top Recommendations Near Poesia (Radius: {radius} km)")
st.map(top_recommendations[['latitude', 'longitude']])

# Display top recommendations clearly
for _, row in top_recommendations.iterrows():
    with st.expander(f"Location {row['location_id']} — Predicted Sales: €{row['predicted_sales']}"):
        st.write(f"""
        **Metrics:**
        - 📈 **Monthly Sales Prediction:** €{row['predicted_sales']}
        - 🚶 **Foot Traffic:** {row['foot_traffic']}/week
        - ⚔️ **Competitor Density:** {row['competitor_density']}
        - 💼 **Demographic Match:** {row['demographic_match']}/10
        - 📊 **Rent Price:** €{row['rent_price']:.2f}/m²
        """)

# Simple Chatbot Interaction
st.markdown("## 💬 Ask AI Assistant")
user_question = st.text_input("Your question:", "Why is this location recommended?")

if user_input:
    st.markdown(f"""
    **AI Advisor says:**  
    "Based on your goal of **{primary_target}**, the top recommended locations within **{radius} km** of Poesia (Gran de Gràcia, 164) have been selected to maximize your desired outcome, factoring in key indicators like foot traffic, revenue potential, and brand alignment."
    """)

st.markdown("---")
st.caption("📌 **Poesia AI Advisor** powered by  Gaston AI © 2025")
