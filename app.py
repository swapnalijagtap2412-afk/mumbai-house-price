
import streamlit as st
import pickle
import json
import numpy as np

# ------------------------------
# Load model
# ------------------------------
with open("Mumbai_house_prices_model.pickle", "rb") as f:
    model = pickle.load(f)

# ------------------------------
# Load columns
# ------------------------------
with open("columns.json", "r") as f:
    data_columns = json.load(f)["data_columns"]

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Mumbai House Price Predictor", layout="centered")

st.title("🏠 Mumbai House Price Prediction App")
st.write("Predict house prices based on location, BHK and area")

# Inputs
bhk = st.number_input("Number of BHK", min_value=1, max_value=10, step=1)
area = st.number_input("Area (in sqft)", min_value=300, max_value=10000, step=50)

locations = data_columns[2:]   # first two are bhk & area
location = st.selectbox("Location", locations)

# ------------------------------
# Prediction button
# ------------------------------
if st.button("Predict Price"):
    x = np.zeros(len(data_columns))
    x[0] = bhk
    x[1] = area

    if location in data_columns:
        loc_index = data_columns.index(location)
        x[loc_index] = 1

    price = model.predict([x])[0]

    st.success(f"💰 Estimated Price: ₹ {round(price, 2)} Lakhs")

