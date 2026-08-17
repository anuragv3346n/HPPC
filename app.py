import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("House price.csv")

# Features and Target
X = df.drop("Price", axis=1)
y = df["Price"]

# Convert categorical columns into numerical columns
X = pd.get_dummies(X, drop_first=True)

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

st.title("🏠 House Price Prediction")

st.subheader("Enter House Details")

# Numeric inputs
area = st.number_input("Area (sq ft)", min_value=100, value=1000)
bedrooms = st.number_input("Bedrooms", min_value=1, value=2)
bathrooms = st.number_input("Bathrooms", min_value=1, value=1)
stories = st.number_input("Stories", min_value=1, value=1)
parking = st.number_input("Parking", min_value=0, value=1)
age = st.number_input("Age", min_value=0, value=5)
rating = st.number_input("Locality Rating", min_value=1, max_value=10, value=5)

city = st.selectbox("City", df["City"].unique())
furnishing = st.selectbox("Furnishing", df["Furnishing"].unique())
main_road = st.selectbox("Main Road", df["Main Road"].unique())
guest_room = st.selectbox("Guest Room", df["Guest Room"].unique())
basement = st.selectbox("Basement", df["Basement"].unique())
water_supply = st.selectbox("Water Supply", df["Water Supply"].unique())
ac = st.selectbox("Air Conditioning", df["Air Conditioning"].unique())
tenant = st.selectbox("Preferred Tenant", df["Preferred Tenant"].unique())

if st.button("Predict Price"):

    input_data = {
        "Area": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Stories": stories,
        "Parking": parking,
        "Age": age,
        "Locality Rating": rating,
        "City": city,
        "Furnishing": furnishing,
        "Main Road": main_road,
        "Guest Room": guest_room,
        "Basement": basement,
        "Water Supply": water_supply,
        "Air Conditioning": ac,
        "Preferred Tenant": tenant
    }

    input_df = pd.DataFrame([input_data])

    # Apply same encoding as training data
    input_df = pd.get_dummies(input_df)

    # Match training columns
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    prediction = model.predict(input_df)

    st.success(f"Predicted House Price: ₹{prediction[0]:,.0f}")