import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --- Load and preprocess the data ---
df = pd.read_csv("data.csv")

# Select useful columns (skip any that are missing)
expected_cols = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
                 'floors', 'waterfront', 'view', 'condition',
                 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated']

available_cols = [col for col in expected_cols if col in df.columns]
df = df[available_cols].dropna()

# Create new features
df['age'] = 2025 - df['yr_built']
df['is_renovated'] = (df['yr_renovated'] > 0).astype(int)

# Define features and target
X = df[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
        'waterfront', 'view', 'condition', 'sqft_above',
        'sqft_basement', 'age', 'is_renovated']]
y = df['price']

# Train model
model = LinearRegression()
model.fit(X, y)

# --- Streamlit UI ---
st.title("🏠 House Price Predictor")
st.markdown("Fill in the details of the house below to estimate its market price.")

# Interactive input sliders
bedrooms = st.slider("Number of Bedrooms", 0, 10, 3)
bathrooms = st.slider("Number of Bathrooms", 0.0, 5.0, 2.0)
sqft_living = st.number_input("Living Area (sqft)", min_value=100, value=1800)
sqft_lot = st.number_input("Lot Size (sqft)", min_value=200, value=4000)
floors = st.slider("Number of Floors", 1, 3, 2)
waterfront = st.selectbox("Waterfront View", [0, 1])
view = st.slider("View Quality (0-4)", 0, 4, 0)
condition = st.slider("House Condition (1-5)", 1, 5, 3)
sqft_above = st.number_input("Above Ground Area (sqft)", min_value=100, value=1500)
sqft_basement = st.number_input("Basement Area (sqft)", min_value=0, value=300)
age = st.number_input("Age of the House", min_value=0, value=25)
is_renovated = st.selectbox("Renovated?", [0, 1])

# Make prediction
input_data = np.array([[bedrooms, bathrooms, sqft_living, sqft_lot, floors,
                        waterfront, view, condition, sqft_above,
                        sqft_basement, age, is_renovated]])

predicted_price = model.predict(input_data)[0]

# Display result
st.subheader(" Predicted House Price:")
st.write(f"₹ {predicted_price:,.0f}")
