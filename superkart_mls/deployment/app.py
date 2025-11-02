import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="jeevachandru/SuperCart_SalesForecast", filename="best_supercartsalesforecast_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("🛒 SuperKart Sales Forecast App")
st.markdown("""
This application predicts **Product Store Sales Total** using product and store attributes.  
It leverages a machine learning model trained on historical sales data to provide accurate forecasts.
""")

# --- User Input Section ---
st.header("Enter Product & Store Details")

product_weight = st.number_input("Product Weight (in kg)", min_value=0.0, value=12.0, step=0.1)
product_sugar = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
product_area = st.number_input("Product Allocated Area (ratio)", min_value=0.0, value=0.05, step=0.001)
product_type = st.selectbox(
    "Product Type",
    ["Frozen Foods", "Dairy", "Canned", "Baking Goods", "Health and Hygiene",
     "Soft Drinks", "Fruits and Vegetables", "Seafood", "Others"]
)
product_mrp = st.number_input("Product MRP (₹)", min_value=50.0, max_value=500.0, value=150.0, step=0.1)
store_year = st.number_input("Store Establishment Year", min_value=1980, max_value=2025, value=2000)
store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
store_city = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
store_type = st.selectbox("Store Type", ["Departmental Store", "Supermarket Type1", "Supermarket Type2", "Food Mart"])

# --- Assemble Input Data ---
input_data = pd.DataFrame([{
    "Product_Weight": product_weight,
    "Product_Sugar_Content": product_sugar,
    "Product_Allocated_Area": product_area,
    "Product_Type": product_type,
    "Product_MRP": product_mrp,
    "Store_Establishment_Year": store_year,
    "Store_Size": store_size,
    "Store_Location_City_Type": store_city,
    "Store_Type": store_type
}])

# --- Predict Sales ---
if st.button("🔮 Predict Sales"):
    prediction = model.predict(input_data)
    st.success(f"💰 Predicted Product Store Sales Total: **₹{prediction[0]:,.2f}**")

# --- Footer ---
st.markdown("---")

  
