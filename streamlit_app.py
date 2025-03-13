import streamlit as st
import pandas as pd
import numpy as np
from faker import Faker
import xgboost as xgb
from sklearn.model_selection import train_test_split
from geopy.distance import geodesic
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Poesia Location Advisor", layout="wide")

# Title and description
st.title("🍽️ Poesia AI-Powered Location Advisor")
st.markdown("### Recommended Store Locations in Barcelona")

fake = Faker()

# Generate synthetic data with realistic addresses
@st.cache_data
def generate_data(num_locations=100):
    data = []
    for _ in range(num_locations):
        lat = np.random.uniform(41.36, 41.42)
        lon = np.random.uniform(2.14, 2.19)
        data.append({
            'address': fake.address().replace("\n", ", "),
            'latitude': lat,
            'longitude': lon,
            'avg_income': np.random.normal(35000, 12000),
            'foot_traffic': np.random.randint(1000, 5000),
            'competitor_density': np.random.randint(0, 10),
            'rent_price': np.random.uniform(20, 80),
            'demographic_match': round(np.random.uniform(6, 10), 1),
            'sentiment_score': round(np.random.uniform(0.5, 0.9), 2)
        })
    return pd.DataFrame(data)

data = generate_data()

# Train model with cached function
@st.cache_resource(show_spinner="Training AI model...")
def train_model(df):
    X = df[['demographic_match', 'foot_traffic', 'competitor_density', 'rent_price', 'sentiment_score']]
    y = (df['foot_traffic'] * 10 +
         df['demographic_match'] * 1000 -
         df['competitor_density'] * 1200 +
         df['sentiment_score'] * 1000 -
         df['rent_price'] * 300).astype(int)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(n_estimators=30, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model(data)

# Sidebar for interaction
with st.sidebar:
    st.image("https://static.wixstatic.com/media/0321ec_ee3b53f39a3f44c9a6fdbd53c903afc5~mv2.png", width=150)
    st.markdown("### 🔍 Configure Your Search")
    radius = st.slider("Search Radius (km)", 0.5, 5.0, 2.0, step=0.1)
    store_type = st.selectbox("Store Type", ["Flagship", "Normal", "Corner"])
    primary_target = st.radio("Primary Goal", ["Revenue", "Brand Awareness", "Foot Traffic"])

# Central location: Poesia, Gran de Gràcia 164
poesia_coords = (41.400893, 2.152517)

# Filter locations within selected radius
data['distance_km'] = data.apply(
    lambda row: geodesic(poesia_coords, (row['latitude'], row['longitude'])).km, axis=1
)
filtered_data = data[data['distance_km'] <= radius].copy()

# Predict sales
features = ['demographic_match', 'foot_traffic', 'competitor_density', 'rent_price', 'sentiment_score']
filtered_data['predicted_sales'] = model.predict(filtered_data[features]).astype(int)

# Scoring based on target
weights = {
    'Revenue': {'sales': 0.6, 'foot_traffic': 0.2, 'demographic_match': 0.2},
    'Brand Awareness': {'sales': 0.2, 'sentiment_score': 0.5, 'foot_traffic': 0.3},
    'Foot Traffic': {'sales': 0.2, 'foot_traffic': 0.6, 'demographic_match': 0.2}
}

w = weights[primary_target]

filtered_data['score'] = (
    filtered_data['predicted_sales'] * w.get('sales', 0) +
    filtered_data['foot_traffic'] * w.get('foot_traffic', 0) * 2 +
    filtered_data['demographic_match'] * w.get('demographic_match', 0) * 1000 +
    filtered_data['sentiment_score'] * w.get('sentiment_score', 0) * 1000
)

top_recommendations = filtered_data.sort_values(by='score', ascending=False).head(5)

# Recommendations on Top (Clearly Displayed)
st.subheader("🏆 Top Recommended Locations")
for _, row in top_recommendations.iterrows():
    st.markdown(f"**📍 {row['address']} — Predicted Monthly Sales: €{row['predicted_sales']:,}**")
    with st.expander("View Details"):
        st.markdown(f"""
        - 🚶 Foot Traffic: **{row['foot_traffic']} per week**
        - ⚔️ Competitor Density: **{row['competitor_density']} nearby**
        - 🧑‍🤝‍🧑 Demographic Match: **{row['demographic_match']}/10**
        - 🏠 Estimated Rent: **€{row['rent_price']:.2f}/m²**
        """)

# Folium Map Integration (Clearly Better UI)
st.subheader("🗺️ Interactive Map of Recommended Locations")
m = folium.Map(location=poesia_coords, zoom_start=14)

# Add central point clearly marked
folium.Marker(poesia_coords, tooltip="Poesia (Gran de Gràcia 164)",
              icon=folium.Icon(color='blue', icon='cutlery')).add_to(m)

# Add recommended points
for idx, row in top_recommendations.iterrows():
    folium.Marker(
        location=(row['latitude'], row['longitude']),
        popup=(f"{row['address']}<br>Sales: €{row['predicted_sales']}"),
        icon=folium.Icon(color='green', icon='ok-sign')
    ).add_to(m)

st_folium(m, width=1000, height=500)

# AI Chatbot section (simple interaction)
st.subheader("💬 Chat with Poesia AI Advisor")
question = st.text_input("Ask about recommended locations:", key="unique_chat_input")

if question:
    st.markdown(f"""
    **AI Advisor:** Based on your primary goal of **{primary_target.lower()}**, the recommended locations within a radius of **{radius} km** from Poesia (Gran de Gràcia, 164) have been carefully selected. These sites optimize for critical metrics like foot traffic, predicted sales, and ideal demographics, enhancing your likelihood of success.
    """)

st.caption("Poesia Location Advisor © 2025 — Powered by AI")
